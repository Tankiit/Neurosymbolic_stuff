import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

class ConceptDiscoveryEngine(nn.Module):
    """
    Discovers polynomial concept combinations from CelebA attributes
    """
    def __init__(self, 
                 base_concepts: List[str], 
                 num_discovered_concepts: int = 10,
                 max_polynomial_degree: int = 2,
                 sparsity_regularization: float = 0.01):
        super().__init__()
        
        self.base_concepts = base_concepts
        self.num_base = len(base_concepts)
        self.num_discovered = num_discovered_concepts
        self.max_degree = max_polynomial_degree
        self.sparsity_reg = sparsity_regularization
        
        # Generate all polynomial terms up to max_degree
        self.polynomial_terms = self._generate_polynomial_terms()
        self.num_terms = len(self.polynomial_terms)
        
        # Learnable coefficients for each discovered concept
        # Shape: (num_discovered_concepts, num_polynomial_terms)
        self.concept_coefficients = nn.Parameter(
            torch.randn(num_discovered_concepts, self.num_terms) * 0.1
        )
        
        # Optional: learnable bias for each discovered concept
        self.concept_bias = nn.Parameter(torch.zeros(num_discovered_concepts))
        
        print(f"Initialized discovery engine with {self.num_terms} polynomial terms")
        print(f"Terms include: {self.polynomial_terms[:10]}...") # Show first 10
    
    def _generate_polynomial_terms(self) -> List[Tuple[int, ...]]:
        """
        Generate all polynomial terms up to max_degree
        Returns list of tuples, where each tuple represents which variables to multiply
        E.g., (0,) means x_0, (0,1) means x_0 * x_1, () means constant term
        """
        terms = [()]  # Constant term
        
        # Linear terms
        for i in range(self.num_base):
            terms.append((i,))
        
        # Higher degree terms
        if self.max_degree >= 2:
            # Quadratic terms: x_i * x_j for all i,j (including i=j)
            for i in range(self.num_base):
                for j in range(i, self.num_base):  # Only upper triangle to avoid duplicates
                    terms.append((i, j))
        
        if self.max_degree >= 3:
            # Cubic terms: x_i * x_j * x_k
            for i in range(self.num_base):
                for j in range(i, self.num_base):
                    for k in range(j, self.num_base):
                        terms.append((i, j, k))
        
        return terms
    
    def evaluate_polynomial_terms(self, base_concept_values: torch.Tensor) -> torch.Tensor:
        """
        Evaluate all polynomial terms for given base concept values
        Args:
            base_concept_values: (batch_size, num_base_concepts)
        Returns:
            polynomial_features: (batch_size, num_polynomial_terms)
        """
        batch_size = base_concept_values.shape[0]
        polynomial_features = torch.zeros(batch_size, self.num_terms, device=base_concept_values.device)
        
        for term_idx, term in enumerate(self.polynomial_terms):
            if len(term) == 0:  # Constant term
                polynomial_features[:, term_idx] = 1.0
            else:
                # Multiply the specified variables
                term_value = torch.ones(batch_size, device=base_concept_values.device)
                for var_idx in term:
                    term_value *= base_concept_values[:, var_idx]
                polynomial_features[:, term_idx] = term_value
        
        return polynomial_features
    
    def forward(self, base_concept_values: torch.Tensor) -> torch.Tensor:
        """
        Compute discovered concepts as polynomial combinations of base concepts
        Args:
            base_concept_values: (batch_size, num_base_concepts)
        Returns:
            discovered_concepts: (batch_size, num_discovered_concepts)
        """
        # Evaluate polynomial terms
        poly_features = self.evaluate_polynomial_terms(base_concept_values)
        
        # Linear combination with learned coefficients
        discovered = torch.matmul(poly_features, self.concept_coefficients.T)
        discovered = discovered + self.concept_bias
        
        # Apply sigmoid to keep in [0,1] range
        return torch.sigmoid(discovered)
    
    def get_concept_interpretation(self, concept_idx: int, threshold: float = 0.1) -> Dict:
        """
        Interpret a discovered concept by showing its polynomial representation
        """
        coeffs = self.concept_coefficients[concept_idx].detach().cpu().numpy()
        bias = self.concept_bias[concept_idx].detach().cpu().item()
        
        interpretation = {
            'bias': bias,
            'terms': []
        }
        
        for term_idx, (coeff, term) in enumerate(zip(coeffs, self.polynomial_terms)):
            if abs(coeff) > threshold:  # Only show significant terms
                term_desc = self._term_to_string(term)
                interpretation['terms'].append({
                    'coefficient': coeff,
                    'term': term_desc,
                    'variables': [self.base_concepts[i] for i in term] if term else ['constant']
                })
        
        # Sort by absolute coefficient value
        interpretation['terms'].sort(key=lambda x: abs(x['coefficient']), reverse=True)
        return interpretation
    
    def _term_to_string(self, term: Tuple[int, ...]) -> str:
        """Convert polynomial term to readable string"""
        if len(term) == 0:
            return "1"
        elif len(term) == 1:
            return self.base_concepts[term[0]]
        else:
            var_names = [self.base_concepts[i] for i in term]
            return " × ".join(var_names)
    
    def sparsity_loss(self) -> torch.Tensor:
        """L1 regularization to encourage sparse representations"""
        return self.sparsity_reg * torch.sum(torch.abs(self.concept_coefficients))
    
    def diversity_loss(self) -> torch.Tensor:
        """Encourage discovered concepts to be different from each other"""
        # Compute pairwise cosine similarities
        coeffs_normalized = F.normalize(self.concept_coefficients, p=2, dim=1)
        similarity_matrix = torch.matmul(coeffs_normalized, coeffs_normalized.T)
        
        # Zero out diagonal (self-similarities)
        mask = torch.eye(self.num_discovered, device=similarity_matrix.device)
        similarity_matrix = similarity_matrix * (1 - mask)
        
        # Penalize high similarities
        return torch.mean(torch.abs(similarity_matrix))


class SupervisedConceptDiscovery(nn.Module):
    """
    Supervised approach: discover concepts that predict specific target attributes
    """
    def __init__(self, base_concepts: List[str], target_attributes: List[str]):
        super().__init__()
        self.base_concepts = base_concepts
        self.target_attributes = target_attributes
        
        # One discovery engine per target attribute
        self.discovery_engines = nn.ModuleDict({
            target: ConceptDiscoveryEngine(base_concepts, num_discovered_concepts=1, max_polynomial_degree=3)
            for target in target_attributes
        })
    
    def forward(self, base_concept_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Discover concepts that predict each target attribute
        """
        discovered = {}
        for target, engine in self.discovery_engines.items():
            discovered[target] = engine(base_concept_values).squeeze(-1)  # Remove last dim
        return discovered
    
    def get_all_interpretations(self) -> Dict[str, Dict]:
        """Get polynomial interpretations for all target attributes"""
        interpretations = {}
        for target, engine in self.discovery_engines.items():
            interpretations[target] = engine.get_concept_interpretation(0)
        return interpretations


class UnsupervisedConceptDiscovery(nn.Module):
    """
    Unsupervised approach: discover latent concepts through reconstruction and clustering
    """
    def __init__(self, 
                 base_concepts: List[str], 
                 num_discovered_concepts: int = 10,
                 bottleneck_dim: int = 5):
        super().__init__()
        self.base_concepts = base_concepts
        self.num_discovered = num_discovered_concepts
        self.bottleneck_dim = bottleneck_dim
        
        # Concept discovery engine
        self.discovery_engine = ConceptDiscoveryEngine(
            base_concepts, 
            num_discovered_concepts=num_discovered_concepts,
            max_polynomial_degree=2
        )
        
        # Reconstruction network: discovered concepts -> base concepts
        self.reconstruction_net = nn.Sequential(
            nn.Linear(num_discovered_concepts, bottleneck_dim),
            nn.ReLU(),
            nn.Linear(bottleneck_dim, len(base_concepts)),
            nn.Sigmoid()
        )
    
    def forward(self, base_concept_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discover concepts and reconstruct base concepts
        """
        discovered = self.discovery_engine(base_concept_values)
        reconstructed = self.reconstruction_net(discovered)
        return discovered, reconstructed
    
    def reconstruction_loss(self, original: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss"""
        return F.mse_loss(reconstructed, original)


def train_supervised_discovery(model: SupervisedConceptDiscovery,
                             dataloader,
                             base_attr_indices: List[int],
                             target_attr_indices: Dict[str, int],
                             num_epochs: int = 20):
    """
    Train supervised concept discovery to predict target attributes
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for images, attributes, _ in dataloader:
            optimizer.zero_grad()
            
            # Extract base concept values (ground truth)
            base_values = attributes[:, base_attr_indices].float()
            
            # Get discovered concept predictions
            discovered = model(base_values)
            
            # Compute prediction loss for each target
            loss = 0
            for target_name, target_idx in target_attr_indices.items():
                target_values = attributes[:, target_idx].float()
                pred_values = discovered[target_name]
                loss += F.binary_cross_entropy(pred_values, target_values)
            
            # Add regularization
            for engine in model.discovery_engines.values():
                loss += engine.sparsity_loss()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Print interpretations every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("\nCurrent concept interpretations:")
            interpretations = model.get_all_interpretations()
            for target, interp in interpretations.items():
                print(f"\n{target}:")
                print(f"  Bias: {interp['bias']:.3f}")
                for term in interp['terms'][:3]:  # Top 3 terms
                    print(f"  {term['coefficient']:.3f} × {term['term']}")


def train_unsupervised_discovery(model: UnsupervisedConceptDiscovery,
                               dataloader,
                               base_attr_indices: List[int],
                               num_epochs: int = 30):
    """
    Train unsupervised concept discovery through reconstruction
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for images, attributes, _ in dataloader:
            optimizer.zero_grad()
            
            # Extract base concept values
            base_values = attributes[:, base_attr_indices].float()
            
            # Discover concepts and reconstruct
            discovered, reconstructed = model(base_values)
            
            # Reconstruction loss
            recon_loss = model.reconstruction_loss(base_values, reconstructed)
            
            # Regularization losses
            sparsity_loss = model.discovery_engine.sparsity_loss()
            diversity_loss = model.discovery_engine.diversity_loss()
            
            total_loss_batch = recon_loss + sparsity_loss + 0.1 * diversity_loss
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        # Print discovered concepts every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("\nDiscovered concepts:")
            for i in range(min(3, model.num_discovered)):  # Show first 3 concepts
                interp = model.discovery_engine.get_concept_interpretation(i)
                print(f"\nConcept {i+1}:")
                print(f"  Bias: {interp['bias']:.3f}")
                for term in interp['terms'][:3]:
                    print(f"  {term['coefficient']:.3f} × {term['term']}")


def analyze_discovered_concepts(model, dataloader, base_attr_indices: List[int], base_concepts: List[str]):
    """
    Analyze what the discovered concepts have learned
    """
    all_base_values = []
    all_discovered = []
    
    # Collect data
    with torch.no_grad():
        for images, attributes, _ in dataloader:
            base_values = attributes[:, base_attr_indices].float()
            
            if isinstance(model, SupervisedConceptDiscovery):
                discovered = model(base_values)
                # Convert dict to tensor
                discovered_tensor = torch.stack(list(discovered.values()), dim=1)
            else:  # UnsupervisedConceptDiscovery
                discovered_tensor, _ = model(base_values)
            
            all_base_values.append(base_values)
            all_discovered.append(discovered_tensor)
    
    base_values = torch.cat(all_base_values, dim=0).numpy()
    discovered_values = torch.cat(all_discovered, dim=0).numpy()
    
    # PCA analysis
    print("\n=== PCA Analysis of Discovered Concepts ===")
    pca = PCA(n_components=min(3, discovered_values.shape[1]))
    discovered_pca = pca.fit_transform(discovered_values)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Correlation analysis
    print("\n=== Correlation with Base Concepts ===")
    correlations = np.corrcoef(base_values.T, discovered_values.T)
    base_discovered_corr = correlations[:len(base_concepts), len(base_concepts):]
    
    for i, concept in enumerate(base_concepts):
        print(f"{concept}: {base_discovered_corr[i]}")
    
    return {
        'base_values': base_values,
        'discovered_values': discovered_values,
        'pca_components': discovered_pca,
        'correlations': base_discovered_corr
    }


# Example usage function
def run_concept_discovery_experiment():
    """
    Complete example of running concept discovery on CelebA
    """
    # Define base and target concepts
    base_concepts = ['Male', 'Young', 'Attractive', 'Smiling', 'Black_Hair', 'Blond_Hair']
    target_concepts = ['Wearing_Lipstick', 'Heavy_Makeup', 'Rosy_Cheeks']
    
    print("Loading CelebA dataset...")
    train_loader, val_loader = load_celeba_dataset()
    
    # Get attribute indices from CelebA
    # CelebA attribute names are lowercase with underscores
    attr_names = train_loader.dataset.attr_names
    
    # Map our concept names to CelebA attribute indices
    celeba_name_map = {
        'Male': 'Male', 
        'Young': 'Young',
        'Attractive': 'Attractive',
        'Smiling': 'Smiling',
        'Black_Hair': 'Black_Hair',
        'Blond_Hair': 'Blond_Hair',
        'Wearing_Lipstick': 'Wearing_Lipstick',
        'Heavy_Makeup': 'Heavy_Makeup',
        'Rosy_Cheeks': 'Rosy_Cheeks'
    }
    
    base_attr_indices = [attr_names.index(celeba_name_map[attr]) for attr in base_concepts]
    target_attr_indices = {target: attr_names.index(celeba_name_map[target]) for target in target_concepts}
    
    print(f"Base attribute indices: {base_attr_indices}")
    print(f"Target attribute indices: {target_attr_indices}")
    
    print("\n=== Supervised Concept Discovery ===")
    supervised_model = SupervisedConceptDiscovery(base_concepts, target_concepts)
    train_supervised_discovery(supervised_model, train_loader, base_attr_indices, target_attr_indices, num_epochs=5)
    
    print("\n=== Unsupervised Concept Discovery ===")
    unsupervised_model = UnsupervisedConceptDiscovery(base_concepts, num_discovered_concepts=5)
    train_unsupervised_discovery(unsupervised_model, train_loader, base_attr_indices, num_epochs=5)
    
    # Analyze results
    print("\n=== Analyzing Discovered Concepts ===")
    results = analyze_discovered_concepts(supervised_model, val_loader, base_attr_indices, base_concepts)
    
    # Print some interpretations
    print("\nSupervised concept interpretations:")
    interpretations = supervised_model.get_all_interpretations()
    for target, interp in interpretations.items():
        print(f"\n{target}:")
        print(f"  Bias: {interp['bias']:.3f}")
        for term in interp['terms'][:5]:  # Top 5 terms
            print(f"  {term['coefficient']:.3f} × {term['term']}")


class CustomCelebADataset(Dataset):
    """Custom CelebA dataset loader that works with the existing file structure"""
    def __init__(self, 
                 root_dir: str,
                 split: str = 'train',
                 transform=None):
        """
        Args:
            root_dir: Directory with the CelebA dataset
            split: 'train', 'valid', or 'test'
            transform: Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Load attributes
        attr_file = os.path.join(root_dir, 'list_attr_celeba.txt')
        self.attr_df = pd.read_csv(attr_file, sep='\s+', skiprows=1)
        
        # Load split information
        split_file = os.path.join(root_dir, 'list_eval_partition.txt')
        split_df = pd.read_csv(split_file, sep='\s+', header=None, names=['image_id', 'split'])
        
        # Map split values (0: train, 1: valid, 2: test)
        split_map = {0: 'train', 1: 'valid', 2: 'test'}
        split_df['split'] = split_df['split'].map(split_map)
        
        # Filter for the requested split
        self.split_df = split_df[split_df['split'] == split]
        
        # Get image paths and attributes for this split
        self.image_ids = self.split_df['image_id'].values
        self.attributes = self.attr_df.loc[self.image_ids].values
        
        # Convert attributes to binary (1 for positive, 0 for negative)
        self.attributes = (self.attributes > 0).astype(np.float32)
        
        # Get attribute names
        self.attr_names = list(self.attr_df.columns)
        
        print(f"Loaded {len(self.image_ids)} images for {split} split")
        print(f"Number of attributes: {len(self.attr_names)}")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, 'img_align_celeba', img_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get attributes
        attributes = torch.FloatTensor(self.attributes[idx])
        
        return image, attributes, idx

def load_celeba_dataset(root_dir: str = "/Users/cril/tanmoy/research/data/img_align_celeba", 
                       batch_size: int = 64,
                       num_workers: int = 4):
    """
    Load the CelebA dataset using our custom dataset loader
    """
    from torchvision import transforms

    # Standard transforms for CelebA
    transform = transforms.Compose([
        transforms.Resize((178, 178)),
        transforms.CenterCrop(148),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Create datasets
    try:
        train_dataset = CustomCelebADataset(
            root_dir=root_dir,
            split='train',
            transform=transform
        )
        
        val_dataset = CustomCelebADataset(
            root_dir=root_dir,
            split='valid',
            transform=transform
        )
    except Exception as e:
        print(f"\nError loading dataset: {str(e)}")
        print(f"Please ensure the dataset is properly organized in: {root_dir}/")
        print("Required files:")
        print("- img_align_celeba/ (directory with images)")
        print("- list_attr_celeba.txt")
        print("- list_eval_partition.txt")
        raise

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    run_concept_discovery_experiment()
