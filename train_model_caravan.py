# %%
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.nn as nn

from info_nce import InfoNCE

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import copy

import lightgbm as lgb  # Import LightGBM

# Read the CSV files
training_data = pd.read_csv('data/training_data_caravan.csv')
validation_data = pd.read_csv('data/validation_data_caravan.csv')
test_data = pd.read_csv('data/test_data_caravan.csv')

# Ensure 'Date' column is datetime
training_data['Date'] = pd.to_datetime(training_data['Date'])
validation_data['Date'] = pd.to_datetime(validation_data['Date'])
test_data['Date'] = pd.to_datetime(test_data['Date'])

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.backends.mps.is_available():  # Check if Apple Silicon is available
    device = torch.device("mps")

# %%
class HydrographDataset(Dataset):
    """
    A PyTorch Dataset class for generating hydrograph data samples for contrastive learning.
    """

    def __init__(self, df, device='cpu'):
        """
        Initializes the HydrographDataset.

        Args:
            df (pd.DataFrame): DataFrame containing the hydrograph data.
            device (str, optional): Device to store tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
        """
        self.device = device

        # Store 'Q' values as a tensor on the specified device
        self.Q_values = torch.tensor(df['Q'].values, dtype=torch.float32).to(self.device)

        # Encode catchment names to unique indices
        catchment_names = df['catchment_name'].unique()
        self.catchment_to_idx = {name: idx for idx, name in enumerate(catchment_names)}

        # Map catchment names in the DataFrame to indices
        df['catchment_idx'] = df['catchment_name'].map(self.catchment_to_idx).astype(int)

        # Create tensors of eligible indices and corresponding catchment indices
        eligible_df = df[df['eligibility']]
        self.eligible_indices = torch.tensor(eligible_df.index.values, dtype=torch.long).to(self.device)
        self.catchment_indices = torch.tensor(eligible_df['catchment_idx'].values, dtype=torch.long).to(self.device)

        # Compute start and end indices for each catchment in eligible_indices
        unique_catchment_indices, counts = torch.unique_consecutive(self.catchment_indices, return_counts=True)
        cumulative_counts = torch.cat([torch.tensor([0], device=self.device), counts.cumsum(0)], dim=0)
        self.catchment_start_indices = cumulative_counts[:-1]
        self.catchment_end_indices = cumulative_counts[1:]
        self.catchment_ids_in_ranges = unique_catchment_indices

        # Store counts as weights for sampling
        self.catchment_counts = counts.float()

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: An arbitrary large number since the data is generated on-the-fly.
        """
        return int(1e9)  # Arbitrary large number for infinite sampling

    def __getitem__(self, idx, k, m, n):
        """
        Generates a batch of data for training.

        Args:
            idx (int): Index of the batch (unused since data is generated on-the-fly).
            k (int): Number of queries per batch.
            m (int): Number of positive samples per query.
            n (int): Number of negative samples per query.

        Returns:
            tuple: (queries_tensor, positives_tensor, negatives_tensor)
                - queries_tensor (torch.Tensor): Tensor of shape [k, 365] containing query hydrographs.
                - positives_tensor (torch.Tensor): Tensor of shape [k, m, 365] containing positive samples.
                - negatives_tensor (torch.Tensor): Tensor of shape [k, n, 365] containing negative samples.
        """
        device = self.device

        # Use all available catchments
        available_catchments = self.catchment_ids_in_ranges
        catchment_start_indices = self.catchment_start_indices
        catchment_end_indices = self.catchment_end_indices

        # Compute weights proportional to the number of eligible indices (counts)
        weights = self.catchment_counts

        # Sample k catchments according to weights
        rand_idx = torch.multinomial(weights, k, replacement=False)

        selected_catchments = available_catchments[rand_idx]
        selected_start_indices = catchment_start_indices[rand_idx]
        selected_end_indices = catchment_end_indices[rand_idx]

        # Initialize tensors to store queries, positives, and negatives
        queries = torch.empty(k, 365, device=device)
        positives = torch.empty(k, m, 365, device=device)
        negatives = torch.empty(k, n, 365, device=device)

        # Iterate over the selected catchments to sample queries, positives, and negatives
        for i in range(k):
            start_idx = selected_start_indices[i]
            end_idx = selected_end_indices[i]
            num_eligible = end_idx - start_idx

            # Get eligible indices for the current catchment
            eligible_indices = self.eligible_indices[start_idx:end_idx]

            # Sample (m + 1) indices for query and positives
            rand_perm = torch.randperm(num_eligible, device=device)[:m + 1]
            sampled_indices = eligible_indices[rand_perm]

            # First index is the query
            query_idx = sampled_indices[0].item()
            query_data = self.Q_values[query_idx:query_idx + 365]
            queries[i] = query_data

            # Remaining indices are positives
            pos_indices = sampled_indices[1:]
            pos_data = torch.stack([self.Q_values[idx.item():idx.item() + 365] for idx in pos_indices])
            positives[i] = pos_data

            # Sample negatives for the current query catchment
            # Exclude the current catchment from negative sampling
            mask_neg = self.catchment_indices != selected_catchments[i]
            negative_indices = self.eligible_indices[mask_neg]
            num_negative_indices = negative_indices.shape[0]

            # Sample n negatives for this query
            neg_indices = negative_indices[torch.randint(0, num_negative_indices, (n,), device=device)]
            neg_data = torch.stack([self.Q_values[idx:idx + 365] for idx in neg_indices])
            negatives[i] = neg_data

        return queries, positives, negatives


# Custom DataLoader
class HydrographDataLoader:
    """
    A simple DataLoader class to generate batches from the HydrographDataset.
    """
    def __init__(self, dataset, k, m, n):
        self.dataset = dataset
        self.k = k
        self.m = m
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        return self.dataset.__getitem__(0, self.k, self.m, self.n)


# Initialize the DataLoader
train_loader = HydrographDataLoader(HydrographDataset(training_data, device=device), k=64, m=1, n=8)

# %%
def create_val_test_hydrographs(data, window_size=365, step_size=365):
    """
    Generating fixed-length hydrographs using a sliding window approach.
    """
    hydrographs = []

    # Group data by catchment name
    for catchment_name, group in data.groupby('catchment_name'):
        # Sort each group by date and reset the index
        group = group.sort_values('Date').reset_index(drop=True)

        # Extract Q values
        Q_values = group['Q'].values

        # Generate sequences using a sliding window approach
        for start_idx in range(0, len(Q_values) - window_size + 1, step_size):
            sequence = Q_values[start_idx:start_idx + window_size]

            # Ensure the sequence length matches the window size and contains no NaN values
            if len(sequence) == window_size and not np.isnan(sequence).any():
                hydrographs.append({
                    'catchment_name': catchment_name,
                    'Q_values': sequence.astype(np.float32)
                })

    return hydrographs

# Process training hydrographs
training_hydrographs = create_val_test_hydrographs(training_data, window_size=365, step_size=50)

# Store training hydrographs in a dictionary
training_hydrographs_dict = {}
for hydrograph in training_hydrographs:
    catchment_name = hydrograph['catchment_name']
    if catchment_name not in training_hydrographs_dict:
        training_hydrographs_dict[catchment_name] = []
    training_hydrographs_dict[catchment_name].append(hydrograph['Q_values'])

# Convert lists to numpy arrays
for catchment_name in training_hydrographs_dict:
    training_hydrographs_dict[catchment_name] = np.array(training_hydrographs_dict[catchment_name])

# Process validation hydrographs
validation_hydrographs = create_val_test_hydrographs(validation_data, window_size=365, step_size=365)

# Store validation hydrographs in a dictionary
validation_hydrographs_dict = {}
for hydrograph in validation_hydrographs:
    catchment_name = hydrograph['catchment_name']
    if catchment_name not in validation_hydrographs_dict:
        validation_hydrographs_dict[catchment_name] = []
    validation_hydrographs_dict[catchment_name].append(hydrograph['Q_values'])

# Convert lists to numpy arrays
for catchment_name in validation_hydrographs_dict:
    validation_hydrographs_dict[catchment_name] = np.array(validation_hydrographs_dict[catchment_name])

# %%

# Define the LSTM-based HydrographEncoder without the projection head
class Hydrograph_encoder(nn.Module):
    def __init__(
        self,
        feature_dim,
        lstm_hidden_dim,
        num_lstm_layers,
        fc_hidden_dims,
        output_dim,
        p=0,
    ):
        super(Hydrograph_encoder, self).__init__()

        self.feature_dim = feature_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers

        self.lstm = nn.LSTM(
            self.feature_dim,
            self.lstm_hidden_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
        )

        # LSTM to latent code
        self.fc_hidden_dims = fc_hidden_dims
        self.fc_layers = []
        self.p = p
        for i in range((len(self.fc_hidden_dims))):
            in_dim = self.lstm_hidden_dim if i == 0 else self.fc_hidden_dims[i - 1]
            out_dim = self.fc_hidden_dims[i]

            self.fc_layers += [nn.Linear(in_dim, out_dim)]
            self.fc_layers += [nn.ReLU()]
            self.fc_layers += [nn.Dropout(p=self.p)]

        self.output_dim = output_dim
        self.fc_layers += [nn.Linear(self.fc_hidden_dims[-1], self.output_dim)]

        self.fc_layers = nn.Sequential(*self.fc_layers)

    def forward(self, inputs):
        out, (_, _) = self.lstm(inputs)
        out = self.fc_layers(out[:, -1, :])

        return out

    def encode(self, query, positive_key, negative_keys):
        
        negative_keys_reshape = negative_keys.contiguous().view(
            -1, negative_keys.size(-2), negative_keys.size(-1)
        )  # batch size * sequence length * feature dim

        combined = torch.cat([query, positive_key, negative_keys_reshape], dim=0)

        out = self.forward(combined)

        query_out = out[0 : query.shape[0], :]
        positive_key_out = out[query.shape[0] : (query.shape[0] * 2), :]
        negative_keys_out = out[query.shape[0] * 2 :, :]

        negative_keys_out = negative_keys_out.contiguous().view(
            negative_keys.size(0), -1, negative_keys_out.size(-1)
        )

        return query_out, positive_key_out, negative_keys_out

# Initialization
model = Hydrograph_encoder(
    feature_dim=1,
    lstm_hidden_dim=128,
    num_lstm_layers=2,
    fc_hidden_dims=[32,16],
    output_dim=16,
    p = 0
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = InfoNCE(negative_mode='paired')  # Ensure your InfoNCE implementation supports unnormalized embeddings

# %%

# Early stopping parameters
patience = 50  # Adjusted patience for early stopping
best_val_accuracy = 0
epochs_no_improve = 0
best_model_state = None

# Initialize a list to store validation results
validation_results = []

# Training loop with early stopping
model.train()
total_loss = 0
num_epochs = 10  # Total number of epochs
for epoch in range(num_epochs):
    for step in range(1000):
        queries, positives, negatives = next(iter(train_loader))

        # Move data to device and reshape
        queries = queries.to(device).unsqueeze(2)  # Shape: (batch_size, 365, 1)
        positives = positives.squeeze(1).to(device).unsqueeze(2)  # Shape: (batch_size, 365, 1)
        negatives = negatives.to(device).unsqueeze(3)  # Shape: (batch_size, num_negatives, 365, 1)

        # Forward pass
        embedding_queries, embedding_positives, embedding_negatives = model.encode(queries, positives, negatives)

        # Compute loss using embeddings without normalization
        loss_value = loss_fn(embedding_queries, embedding_positives, embedding_negatives)
        total_loss += loss_value.item()

        # Backward pass
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f'Epoch {epoch + 1}, Step {step + 1}, Loss {total_loss / 10:.4f}')
            total_loss = 0

    # Evaluation on validation set
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        # Set the model to evaluation mode and freeze its parameters
        for param in model.parameters():
            param.requires_grad = False

        # Function to extract embeddings and labels
        def extract_embeddings_and_labels(hydrographs_dict, catchment_to_idx, device):
            embeddings_list = []
            labels_list = []
            for catchment_name, sequences in hydrographs_dict.items():
                sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device).unsqueeze(2)
                embeddings = model(sequences_tensor).cpu().numpy()  # Get embeddings
                embeddings_list.append(embeddings)
                # Use catchment indices as labels
                catchment_idx = catchment_to_idx.get(catchment_name, -1)
                labels = np.full(len(embeddings), catchment_idx, dtype=np.int64)
                labels_list.append(labels)
            # Concatenate all embeddings and labels
            embeddings = np.concatenate(embeddings_list, axis=0)
            labels = np.concatenate(labels_list, axis=0)
            # Remove any entries with invalid catchment indices
            valid_indices = labels != -1
            embeddings = embeddings[valid_indices]
            labels = labels[valid_indices]
            return embeddings, labels

        # Extract embeddings and labels for training data
        train_embeddings, train_labels = extract_embeddings_and_labels(
            training_hydrographs_dict, train_loader.dataset.catchment_to_idx, device
        )

        # Extract embeddings and labels for validation data
        val_embeddings, val_labels = extract_embeddings_and_labels(
            validation_hydrographs_dict, train_loader.dataset.catchment_to_idx, device
        )

        # Normalize embeddings (important for LightGBM)
        scaler = StandardScaler()
        train_embeddings = scaler.fit_transform(train_embeddings)
        val_embeddings = scaler.transform(val_embeddings)
        
        # Check for NaN or constant values in embeddings
        assert not np.isnan(train_embeddings).any(), "Train embeddings contain NaN values!"
        assert not np.isnan(val_embeddings).any(), "Validation embeddings contain NaN values!"
        assert np.var(train_embeddings, axis=0).all() > 0, "Train embeddings have low or zero variance!"
        assert np.var(val_embeddings, axis=0).all() > 0, "Validation embeddings have low or zero variance!"
        
        # Initialize and train the LightGBM classifier
        lightgbm_classifier = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=len(train_loader.dataset.catchment_to_idx),
            boosting_type='gbdt',
            learning_rate=0.05,       # Lower learning rate
            num_leaves=31,            # Smaller tree complexity
            max_depth=7,              # Limit tree depth
            min_child_samples=20,     # Require more samples per leaf
            reg_alpha=0.1,            # L1 regularization
            reg_lambda=0.1,           # L2 regularization
            n_estimators=1000,        # High initial number of trees
            n_jobs=-1,   # Use all CPU cores 
            verbose=-1 # Hide LightGBM output
        )
        
        # Train on the training embeddings
        lightgbm_classifier.fit(train_embeddings, train_labels)
        
        # Predict on the validation embeddings
        val_pred_labels = lightgbm_classifier.predict(val_embeddings)
        
        # Calculate accuracy
        val_accuracy = accuracy_score(val_labels, val_pred_labels)
        print(f'Validation Accuracy after Epoch {epoch + 1}: {val_accuracy * 100:.2f}%')
                # Early stopping logic
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break

        # Unfreeze the model parameters
        for param in model.parameters():
            param.requires_grad = True

        model.train()  # Set model back to training mode

# Save validation results to a CSV file
validation_results_df = pd.DataFrame(validation_results)
validation_results_df.to_csv('validation_results.csv', index=False)

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Save the model
torch.save(model.state_dict(), 'trained_model_camels_de.pth')

# Save the CPU version of the model
torch.save(model.cpu().state_dict(), 'trained_model_camels_de_cpu.pth')
# %%