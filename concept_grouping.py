"""
Universal Rule Discovery System for Popular CBM Datasets
Automatically discover concept groups and constraints across different domains
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re

# ============================================================================
# 1. DATASET CONFIGURATIONS
# ============================================================================

@dataclass
class DatasetConfig:
    """Configuration for different CBM datasets"""
    name: str
    concept_names: List[str]
    concept_descriptions: Dict[str, str]
    domain_type: str  # 'visual', 'medical', 'biological', 'behavioral'
    has_hierarchical_structure: bool
    typical_constraints: List[str]
    data_loader_fn: Optional[str] = None

class DatasetRegistry:
    """Registry of popular CBM datasets with their configurations"""
    
    @staticmethod
    def get_celeba_config() -> DatasetConfig:
        """CelebA face attributes dataset"""
        concept_names = [
            'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
            'Receding_Hairline', 'Straight_Hair', 'Wavy_Hair', 'Mustache', 'Goatee',
            'Sideburns', '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Bushy_Eyebrows',
            'Bags_Under_Eyes', 'Big_Lips', 'Big_Nose', 'Pointy_Nose', 'High_Cheekbones',
            'Rosy_Cheeks', 'Oval_Face', 'Double_Chin', 'Chubby', 'Narrow_Eyes',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
            'Wearing_Necktie', 'Eyeglasses', 'Mouth_Slightly_Open', 'Smiling',
            'Male', 'Young', 'Attractive'
        ]
        
        descriptions = {
            'Bald': 'Person has no hair or very little hair',
            'Black_Hair': 'Person has black colored hair',
            'Male': 'Person appears to be male',
            'Mustache': 'Person has facial hair above upper lip',
            'Wearing_Lipstick': 'Person is wearing lipstick',
            'Smiling': 'Person has a smile expression'
        }
        
        return DatasetConfig(
            name='CelebA',
            concept_names=concept_names,
            concept_descriptions=descriptions,
            domain_type='visual',
            has_hierarchical_structure=True,
            typical_constraints=['hair_color_exclusion', 'gender_implications', 'accessory_combinations']
        )
    
    @staticmethod
    def get_cub_config() -> DatasetConfig:
        """CUB-200-2011 birds dataset"""
        concept_names = [
            # Bill features
            'bill_shape_hooked', 'bill_shape_needle', 'bill_shape_spatulate', 'bill_shape_all_purpose',
            'bill_length_short', 'bill_length_medium', 'bill_length_long',
            'bill_color_black', 'bill_color_white', 'bill_color_red', 'bill_color_brown',
            
            # Wing features  
            'wing_color_black', 'wing_color_white', 'wing_color_brown', 'wing_color_gray',
            'wing_color_blue', 'wing_color_red', 'wing_pattern_solid', 'wing_pattern_spotted',
            'wing_pattern_striped', 'wing_pattern_multi_colored',
            
            # Size and shape
            'size_very_small', 'size_small', 'size_medium', 'size_large',
            'shape_perching_like', 'shape_upright_perching_water_like', 'shape_chicken_like',
            'shape_long_legged_like', 'shape_hawks_like', 'shape_pigeon_like',
            
            # Behavior and habitat
            'habitat_water', 'habitat_woodland', 'habitat_marsh', 'habitat_ocean',
            'behavior_ground_foraging', 'behavior_water_diving', 'behavior_soaring'
        ]
        
        descriptions = {
            'bill_shape_hooked': 'Bird has a hooked/curved bill for tearing meat',
            'habitat_water': 'Bird typically lives near or in water',
            'behavior_soaring': 'Bird soars high in the sky while flying'
        }
        
        return DatasetConfig(
            name='CUB',
            concept_names=concept_names,
            concept_descriptions=descriptions,
            domain_type='biological',
            has_hierarchical_structure=True,
            typical_constraints=['anatomy_consistency', 'habitat_behavior_correlation', 'size_shape_constraints']
        )
    
    @staticmethod
    def get_awa_config() -> DatasetConfig:
        """Animals with Attributes dataset"""
        concept_names = [
            # Colors
            'black', 'white', 'blue', 'brown', 'gray', 'orange', 'red', 'yellow',
            
            # Patterns
            'patches', 'spots', 'stripes',
            
            # Physical attributes
            'furry', 'hairless', 'toughskin', 'big', 'small', 'bulbous', 'lean',
            
            # Body parts
            'flippers', 'hands', 'hooves', 'pads', 'paws', 'longleg', 'longneck', 'tail',
            'chewteeth', 'meatteeth', 'buckteeth', 'strainteeth', 'horns', 'claws', 'tusks',
            
            # Abilities and behaviors
            'forager', 'grazer', 'hunter', 'scavenger', 'skimmer', 'stalker',
            'walk', 'swim', 'tunnels', 'tree', 'group', 'solitary', 'nestspot',
            'domestic', 'fierce', 'timid', 'smart', 'active', 'inactive', 'nocturnal',
            
            # Habitat
            'newworld', 'oldworld', 'arctic', 'coastal', 'desert', 'bush', 'plains',
            'forest', 'fields', 'jungle', 'mountains', 'ocean', 'ground', 'water', 'cave',
            
            # Diet
            'vegetation', 'fish', 'plankton', 'insects', 'meat'
        ]
        
        descriptions = {
            'furry': 'Animal has fur covering its body',
            'hunter': 'Animal actively hunts other animals for food',
            'swim': 'Animal is capable of swimming'
        }
        
        return DatasetConfig(
            name='AwA',
            concept_names=concept_names,
            concept_descriptions=descriptions,
            domain_type='biological',
            has_hierarchical_structure=True,
            typical_constraints=['anatomy_physics', 'diet_behavior', 'habitat_adaptation']
        )
    
    @staticmethod
    def get_chexpert_config() -> DatasetConfig:
        """CheXpert chest X-ray dataset"""
        concept_names = [
            # Pathological findings
            'atelectasis', 'cardiomegaly', 'consolidation', 'edema', 'enlarged_cardiomediastinum',
            'fracture', 'lung_lesion', 'lung_opacity', 'pleural_effusion', 'pleural_other',
            'pneumonia', 'pneumothorax', 'support_devices',
            
            # Anatomical regions
            'heart_size_normal', 'heart_size_enlarged', 'lung_fields_clear', 'lung_fields_abnormal',
            'costophrenic_angles_sharp', 'costophrenic_angles_blunted', 'mediastinum_normal',
            'mediastinum_widened', 'diaphragm_normal', 'diaphragm_elevated',
            
            # Technical quality
            'image_quality_adequate', 'positioning_adequate', 'inspiration_adequate',
            'penetration_adequate', 'rotation_absent', 'artifacts_absent'
        ]
        
        descriptions = {
            'cardiomegaly': 'Enlarged heart visible on chest X-ray',
            'pneumothorax': 'Collapsed lung due to air in pleural space',
            'pleural_effusion': 'Fluid accumulation in pleural space around lungs'
        }
        
        return DatasetConfig(
            name='CheXpert',
            concept_names=concept_names,
            concept_descriptions=descriptions,
            domain_type='medical',
            has_hierarchical_structure=True,
            typical_constraints=['anatomical_consistency', 'pathological_correlations', 'technical_quality']
        )
    
    @staticmethod
    def get_skin_lesion_config() -> DatasetConfig:
        """Skin lesion (HAM10000/ISIC) dataset"""
        concept_names = [
            # ABCDE criteria
            'asymmetry_present', 'border_irregular', 'color_variegated', 'diameter_large', 'evolving',
            
            # Colors
            'color_black', 'color_blue', 'color_brown', 'color_gray', 'color_red', 'color_white',
            'color_pink', 'color_tan', 'color_light_brown', 'color_dark_brown',
            
            # Patterns and structures
            'pattern_network', 'pattern_dots_globules', 'pattern_streaks', 'pattern_blotches',
            'pattern_regression', 'pattern_blue_whitish_veil',
            
            # Surface features
            'surface_smooth', 'surface_rough', 'surface_scaly', 'surface_keratotic',
            'ulceration_present', 'bleeding_present', 'crusting_present',
            
            # Dermoscopic features
            'pigment_network_typical', 'pigment_network_atypical', 'dots_regular', 'dots_irregular',
            'vessels_hairpin', 'vessels_dotted', 'vessels_linear_irregular',
            
            # Location and size
            'location_sun_exposed', 'location_sun_protected', 'size_small', 'size_medium', 'size_large',
            
            # Clinical features
            'raised_lesion', 'flat_lesion', 'nodular_component', 'inflammation_present'
        ]
        
        descriptions = {
            'asymmetry_present': 'Lesion is asymmetrical in shape',
            'border_irregular': 'Lesion has irregular or poorly defined borders',
            'color_variegated': 'Lesion shows multiple colors or color variation'
        }
        
        return DatasetConfig(
            name='SkinLesion',
            concept_names=concept_names,
            concept_descriptions=descriptions,
            domain_type='medical',
            has_hierarchical_structure=True,
            typical_constraints=['abcde_criteria', 'color_pattern_consistency', 'clinical_correlation']
        )

# ============================================================================
# 2. DOMAIN-AWARE RULE DISCOVERY
# ============================================================================

class DomainAwareRuleDiscovery:
    """Domain-aware rule discovery that adapts to different dataset types"""
    
    def __init__(self, dataset_config: DatasetConfig):
        self.config = dataset_config
        self.concept_names = dataset_config.concept_names
        self.domain_type = dataset_config.domain_type
        
        # Domain-specific rule templates
        self.rule_templates = self._get_domain_rule_templates()
        
    def _get_domain_rule_templates(self) -> Dict[str, List[str]]:
        """Get rule templates specific to domain type"""
        
        if self.domain_type == 'visual':
            return {
                'color_exclusion': ['color_*', 'hair_color_*'],
                'gender_implications': ['facial_hair_*', 'makeup_*'],
                'physical_consistency': ['size_*', 'shape_*'],
                'accessory_combinations': ['wearing_*']
            }
        
        elif self.domain_type == 'biological':
            return {
                'anatomy_physics': ['size_*', 'body_part_*'],
                'habitat_adaptation': ['habitat_*', 'behavior_*'],
                'diet_behavior': ['diet_*', 'teeth_*', 'behavior_*'],
                'locomotion_anatomy': ['movement_*', 'body_structure_*']
            }
        
        elif self.domain_type == 'medical':
            return {
                'anatomical_consistency': ['organ_*', 'region_*'],
                'pathological_correlations': ['disease_*', 'symptom_*'],
                'severity_progression': ['mild_*', 'moderate_*', 'severe_*'],
                'diagnostic_criteria': ['criterion_*', 'feature_*']
            }
        
        else:
            return {
                'general_exclusion': ['*'],
                'general_implication': ['*'],
                'general_correlation': ['*']
            }
    
    def discover_domain_specific_groups(self, concept_data: np.ndarray) -> Dict[str, List[str]]:
        """Discover concept groups using domain knowledge"""
        
        groups = {}
        
        if self.domain_type == 'visual':
            groups.update(self._discover_visual_groups(concept_data))
        elif self.domain_type == 'biological':
            groups.update(self._discover_biological_groups(concept_data))
        elif self.domain_type == 'medical':
            groups.update(self._discover_medical_groups(concept_data))
        
        # Always add correlation-based groups
        groups.update(self._discover_correlation_groups(concept_data))
        
        return groups
    
    def _discover_visual_groups(self, concept_data: np.ndarray) -> Dict[str, List[str]]:
        """Discover groups for visual datasets (CelebA, etc.)"""
        groups = {}
        
        # Hair-related concepts
        hair_concepts = [c for c in self.concept_names 
                        if any(keyword in c.lower() for keyword in ['hair', 'bald', 'bang'])]
        if len(hair_concepts) > 1:
            groups['hair_features'] = hair_concepts
        
        # Color concepts
        color_concepts = [c for c in self.concept_names 
                         if any(keyword in c.lower() for keyword in ['black', 'white', 'brown', 'blond', 'gray'])]
        if len(color_concepts) > 1:
            groups['color_features'] = color_concepts
        
        # Facial features
        facial_concepts = [c for c in self.concept_names 
                          if any(keyword in c.lower() for keyword in ['nose', 'lips', 'eyes', 'cheek', 'chin'])]
        if len(facial_concepts) > 1:
            groups['facial_features'] = facial_concepts
        
        # Accessories
        accessory_concepts = [c for c in self.concept_names 
                             if 'wearing' in c.lower() or 'glasses' in c.lower()]
        if len(accessory_concepts) > 1:
            groups['accessories'] = accessory_concepts
        
        return groups
    
    def _discover_biological_groups(self, concept_data: np.ndarray) -> Dict[str, List[str]]:
        """Discover groups for biological datasets (CUB, AwA)"""
        groups = {}
        
        # Anatomical features
        anatomy_keywords = ['bill', 'wing', 'tail', 'leg', 'head', 'body', 'feet', 'paw', 'claw']
        for keyword in anatomy_keywords:
            related_concepts = [c for c in self.concept_names if keyword in c.lower()]
            if len(related_concepts) > 1:
                groups[f'{keyword}_features'] = related_concepts
        
        # Color and pattern
        color_concepts = [c for c in self.concept_names 
                         if any(color in c.lower() for color in ['black', 'white', 'brown', 'red', 'blue', 'gray'])]
        if len(color_concepts) > 1:
            groups['color_features'] = color_concepts
        
        pattern_concepts = [c for c in self.concept_names 
                           if any(pattern in c.lower() for pattern in ['stripe', 'spot', 'solid', 'pattern'])]
        if len(pattern_concepts) > 1:
            groups['pattern_features'] = pattern_concepts
        
        # Size categories
        size_concepts = [c for c in self.concept_names 
                        if any(size in c.lower() for size in ['small', 'large', 'big', 'tiny', 'huge'])]
        if len(size_concepts) > 1:
            groups['size_features'] = size_concepts
        
        # Habitat
        habitat_concepts = [c for c in self.concept_names 
                           if any(habitat in c.lower() for habitat in ['water', 'forest', 'desert', 'ocean', 'habitat'])]
        if len(habitat_concepts) > 1:
            groups['habitat_features'] = habitat_concepts
        
        # Behavior
        behavior_concepts = [c for c in self.concept_names 
                            if any(behavior in c.lower() for behavior in ['hunt', 'forag', 'graz', 'behavior', 'swim', 'fly'])]
        if len(behavior_concepts) > 1:
            groups['behavior_features'] = behavior_concepts
        
        return groups
    
    def _discover_medical_groups(self, concept_data: np.ndarray) -> Dict[str, List[str]]:
        """Discover groups for medical datasets"""
        groups = {}
        
        # Pathological findings
        pathology_concepts = [c for c in self.concept_names 
                             if any(keyword in c.lower() for keyword in 
                                   ['pneumonia', 'edema', 'effusion', 'atelectasis', 'fracture'])]
        if len(pathology_concepts) > 1:
            groups['pathological_findings'] = pathology_concepts
        
        # Anatomical regions
        anatomy_concepts = [c for c in self.concept_names 
                           if any(keyword in c.lower() for keyword in 
                                 ['heart', 'lung', 'mediastin', 'diaphragm', 'costophrenic'])]
        if len(anatomy_concepts) > 1:
            groups['anatomical_regions'] = anatomy_concepts
        
        # Image quality
        quality_concepts = [c for c in self.concept_names 
                           if any(keyword in c.lower() for keyword in 
                                 ['quality', 'position', 'rotation', 'artifact', 'adequate'])]
        if len(quality_concepts) > 1:
            groups['image_quality'] = quality_concepts
        
        # For skin lesions - ABCDE criteria
        if 'asymmetry' in str(self.concept_names).lower():
            abcde_concepts = [c for c in self.concept_names 
                             if any(keyword in c.lower() for keyword in 
                                   ['asymmetry', 'border', 'color', 'diameter', 'evolving'])]
            if len(abcde_concepts) > 1:
                groups['abcde_criteria'] = abcde_concepts
        
        return groups
    
    def _discover_correlation_groups(self, concept_data: np.ndarray) -> Dict[str, List[str]]:
        """Discover groups based on statistical correlation"""
        from sklearn.cluster import AgglomerativeClustering
        
        # Compute correlation matrix
        corr_matrix = np.abs(np.corrcoef(concept_data.T))
        
        # Convert to distance matrix
        distance_matrix = 1 - corr_matrix
        
        # Hierarchical clustering
        n_clusters = min(8, len(self.concept_names) // 4)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage='ward',
            metric='precomputed' if np.all(np.diag(distance_matrix) == 0) else 'euclidean'
        )
        
        if np.all(np.diag(distance_matrix) == 0):
            cluster_labels = clustering.fit_predict(distance_matrix)
        else:
            cluster_labels = clustering.fit_predict(concept_data.T)
        
        # Create groups
        groups = {}
        for cluster_id in range(n_clusters):
            cluster_concepts = [self.concept_names[i] for i in range(len(cluster_labels))
                              if cluster_labels[i] == cluster_id]
            if len(cluster_concepts) > 1:
                groups[f'correlation_group_{cluster_id}'] = cluster_concepts
        
        return groups
    
    def discover_domain_specific_constraints(self, concept_data: np.ndarray) -> List[Dict]:
        """Discover constraints using domain-specific knowledge"""
        
        constraints = []
        
        if self.domain_type == 'visual':
            constraints.extend(self._discover_visual_constraints(concept_data))
        elif self.domain_type == 'biological':
            constraints.extend(self._discover_biological_constraints(concept_data))
        elif self.domain_type == 'medical':
            constraints.extend(self._discover_medical_constraints(concept_data))
        
        # Add general statistical constraints
        constraints.extend(self._discover_statistical_constraints(concept_data))
        
        return constraints
    
    def _discover_visual_constraints(self, concept_data: np.ndarray) -> List[Dict]:
        """Discover constraints for visual datasets"""
        constraints = []
        
        # Hair color mutual exclusion
        hair_colors = [c for c in self.concept_names 
                      if any(color in c.lower() for color in ['black_hair', 'blond_hair', 'brown_hair', 'gray_hair'])]
        if len(hair_colors) > 1:
            constraints.append({
                'type': 'at_most_one',
                'variables': hair_colors,
                'reasoning': 'Person typically has one dominant hair color',
                'domain_specific': True
            })
        
        # Gender-facial hair implications
        facial_hair_concepts = [c for c in self.concept_names 
                               if any(keyword in c.lower() for keyword in ['mustache', 'goatee', 'sideburn', 'shadow'])]
        if facial_hair_concepts and 'Male' in self.concept_names:
            for concept in facial_hair_concepts:
                constraints.append({
                    'type': 'implies',
                    'antecedent': concept,
                    'consequent': 'Male',
                    'reasoning': 'Facial hair typically implies male gender',
                    'domain_specific': True
                })
        
        # Bald and hair style conflicts
        if 'Bald' in self.concept_names:
            hair_styles = [c for c in self.concept_names 
                          if any(style in c.lower() for style in ['bangs', 'wavy', 'straight']) and 'hair' in c.lower()]
            for style in hair_styles:
                constraints.append({
                    'type': 'not_both',
                    'variables': ['Bald', style],
                    'reasoning': 'Bald people cannot have hair styles',
                    'domain_specific': True
                })
        
        return constraints
    
    def _discover_biological_constraints(self, concept_data: np.ndarray) -> List[Dict]:
        """Discover constraints for biological datasets"""
        constraints = []
        
        # Size constraints
        size_concepts = [c for c in self.concept_names 
                        if any(size in c.lower() for size in ['small', 'large', 'big', 'tiny'])]
        if len(size_concepts) > 1:
            # Find opposing sizes
            small_concepts = [c for c in size_concepts if 'small' in c.lower() or 'tiny' in c.lower()]
            large_concepts = [c for c in size_concepts if 'large' in c.lower() or 'big' in c.lower()]
            
            for small in small_concepts:
                for large in large_concepts:
                    constraints.append({
                        'type': 'not_both',
                        'variables': [small, large],
                        'reasoning': 'Cannot be both small and large simultaneously',
                        'domain_specific': True
                    })
        
        # Habitat-behavior correlations
        water_concepts = [c for c in self.concept_names if 'water' in c.lower()]
        swim_concepts = [c for c in self.concept_names if 'swim' in c.lower()]
        
        for water in water_concepts:
            for swim in swim_concepts:
                constraints.append({
                    'type': 'implies',
                    'antecedent': water,
                    'consequent': swim,
                    'reasoning': 'Water habitat implies swimming ability',
                    'domain_specific': True
                })
        
        # Anatomy-function relationships
        if 'bill_shape_hooked' in self.concept_names and 'shape_hawks_like' in self.concept_names:
            constraints.append({
                'type': 'implies',
                'antecedent': 'bill_shape_hooked',
                'consequent': 'shape_hawks_like',
                'reasoning': 'Hooked bills are typical of hawk-like birds',
                'domain_specific': True
            })
        
        return constraints
    
    def _discover_medical_constraints(self, concept_data: np.ndarray) -> List[Dict]:
        """Discover constraints for medical datasets"""
        constraints = []
        
        # Pathological correlations
        if 'cardiomegaly' in self.concept_names and 'heart_size_enlarged' in self.concept_names:
            constraints.append({
                'type': 'implies',
                'antecedent': 'cardiomegaly',
                'consequent': 'heart_size_enlarged',
                'reasoning': 'Cardiomegaly implies enlarged heart size',
                'domain_specific': True
            })
        
        # Anatomical impossibilities
        pneumo_concepts = [c for c in self.concept_names if 'pneumothorax' in c.lower()]
        effusion_concepts = [c for c in self.concept_names if 'pleural_effusion' in c.lower()]
        
        for pneumo in pneumo_concepts:
            for effusion in effusion_concepts:
                constraints.append({
                    'type': 'not_both',
                    'variables': [pneumo, effusion],
                    'reasoning': 'Pneumothorax and pleural effusion rarely coexist',
                    'domain_specific': True
                })
        
        # ABCDE criteria for skin lesions
        if 'asymmetry_present' in self.concept_names:
            abcde_features = ['asymmetry_present', 'border_irregular', 'color_variegated', 'diameter_large']
            present_features = [f for f in abcde_features if f in self.concept_names]
            
            if len(present_features) >= 3:
                constraints.append({
                    'type': 'and_implies',
                    'antecedents': present_features[:3],
                    'consequent': 'malignant_suspicious' if 'malignant_suspicious' in self.concept_names else present_features[-1],
                    'reasoning': 'Multiple ABCDE criteria suggest malignancy',
                    'domain_specific': True
                })
        
        return constraints
    
    def _discover_statistical_constraints(self, concept_data: np.ndarray) -> List[Dict]:
        """Discover general statistical constraints"""
        constraints = []
        
        binary_data = (concept_data > 0.5).astype(int)
        n_samples = len(binary_data)
        
        # Mutual exclusion (low co-occurrence)
        for i in range(len(self.concept_names)):
            for j in range(i + 1, len(self.concept_names)):
                both_count = np.sum(binary_data[:, i] & binary_data[:, j])
                either_count = np.sum(binary_data[:, i] | binary_data[:, j])
                
                if either_count > 0:
                    co_occurrence_rate = both_count / either_count
                    
                    if co_occurrence_rate < 0.05 and either_count > n_samples * 0.1:
                        constraints.append({
                            'type': 'not_both',
                            'variables': [self.concept_names[i], self.concept_names[j]],
                            'confidence': 1 - co_occurrence_rate,
                            'support': either_count / n_samples,
                            'reasoning': f'Low co-occurrence rate: {co_occurrence_rate:.3f}',
                            'domain_specific': False
                        })
        
        # Strong implications
        for i in range(len(self.concept_names)):
            for j in range(len(self.concept_names)):
                if i == j:
                    continue
                
                a_count = np.sum(binary_data[:, i])
                ab_count = np.sum(binary_data[:, i] & binary_data[:, j])
                
                if a_count > 0:
                    confidence = ab_count / a_count
                    support = a_count / n_samples
                    
                    if confidence > 0.8 and support > 0.1:
                        constraints.append({
                            'type': 'implies',
                            'antecedent': self.concept_names[i],
                            'consequent': self.concept_names[j],
                            'confidence': confidence,
                            'support': support,
                            'reasoning': f'High confidence implication: {confidence:.3f}',
                            'domain_specific': False
                        })
        
        return constraints

# ============================================================================
# 3. UNIVERSAL DISCOVERY SYSTEM
# ============================================================================

class UniversalCBMDiscovery:
    """Universal system that works across all CBM datasets"""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.config = self._get_dataset_config(dataset_name)
        self.domain_discovery = DomainAwareRuleDiscovery(self.config)
    
    def _get_dataset_config(self, dataset_name: str) -> DatasetConfig:
        """Get configuration for specified dataset"""
        registry = DatasetRegistry()
        
        if dataset_name.lower() == 'celeba':
            return registry.get_celeba_config()
        elif dataset_name.lower() == 'cub':
            return registry.get_cub_config()
        elif dataset_name.lower() == 'awa':
            return registry.get_awa_config()
        elif dataset_name.lower() == 'chexpert':
            return registry.get_chexpert_config()
        elif dataset_name.lower() == 'skinlesion':
            return registry.get_skin_lesion_config()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def discover_all_rules(self, concept_data: np.ndarray) -> Tuple[Dict[str, List[str]], List[Dict]]:
        """
        Discover all concept groups and constraints for the dataset
        
        Args:
            concept_data: (n_samples, n_concepts) concept annotation data
            
        Returns:
            groups: Discovered concept groups
            constraints: Discovered constraints with confidence scores
        """
        
        print(f"DISCOVERING RULES FOR {self.dataset_name.upper()} DATASET")
        print("=" * 60)
        print(f"Domain type: {self.config.domain_type}")
        print(f"Number of concepts: {len(self.config.concept_names)}")
        print(f"Data shape: {concept_data.shape}")
        
        # Discover concept groups
        print(f"\n1. DISCOVERING CONCEPT GROUPS")
        print("-" * 40)
        groups = self.domain_discovery.discover_domain_specific_groups(concept_data)
        
        print(f"Found {len(groups)} concept groups:")
        for group_name, concepts in groups.items():
            print(f"  {group_name}: {len(concepts)} concepts")
            print(f"    {concepts[:3]}{'...' if len(concepts) > 3 else ''}")
        
        # Discover constraints
        print(f"\n2. DISCOVERING CONSTRAINTS")
        print("-" * 40)
        constraints = self.domain_discovery.discover_domain_specific_constraints(concept_data)
        
        # Filter and rank constraints
        filtered_constraints = self._filter_and_rank_constraints(constraints)
        
        print(f"Found {len(filtered_constraints)} high-quality constraints:")
        for i, constraint in enumerate(filtered_constraints[:5]):
            constraint_type = constraint['type']
            if constraint_type == 'not_both':
                print(f"  {i+1}. NOT_BOTH: {constraint['variables']}")
            elif constraint_type == 'implies':
                print(f"  {i+1}. IMPLIES: {constraint['antecedent']} → {constraint['consequent']}")
            elif constraint_type == 'at_most_one':
                print(f"  {i+1}. AT_MOST_ONE: {constraint['variables']}")
            
            if 'confidence' in constraint:
                print(f"      Confidence: {constraint['confidence']:.3f}")
        
        print(f"\n3. SUMMARY")
        print("-" * 40)
        print(f"✓ {len(groups)} concept groups discovered")
        print(f"✓ {len(filtered_constraints)} constraints discovered")
        print(f"✓ {sum(1 for c in filtered_constraints if c.get('domain_specific', False))} domain-specific rules")
        print(f"✓ {sum(1 for c in filtered_constraints if not c.get('domain_specific', False))} statistical rules")
        
        return groups, filtered_constraints
    
    def _filter_and_rank_constraints(self, constraints: List[Dict]) -> List[Dict]:
        """Filter and rank constraints by quality and relevance"""
        
        # Remove duplicates
        seen = set()
        filtered = []
        
        for constraint in constraints:
            # Create signature for deduplication
            if constraint['type'] in ['not_both', 'at_most_one']:
                signature = (constraint['type'], tuple(sorted(constraint['variables'])))
            elif constraint['type'] == 'implies':
                signature = (constraint['type'], constraint['antecedent'], constraint['consequent'])
            else:
                signature = str(constraint)
            
            if signature not in seen:
                seen.add(signature)
                filtered.append(constraint)
        
        # Rank by quality score
        def quality_score(constraint):
            score = 0
            
            # Domain-specific constraints get higher score
            if constraint.get('domain_specific', False):
                score += 2
            
            # Confidence-based scoring
            if 'confidence' in constraint:
                score += constraint['confidence']
            
            # Support-based scoring
            if 'support' in constraint:
                score += constraint['support'] * 0.5
            
            return score
        
        filtered.sort(key=quality_score, reverse=True)
        return filtered[:30]  # Return top 30 constraints

def demonstrate_universal_discovery():
    """Demonstrate universal discovery across different datasets"""
    
    print("UNIVERSAL CBM RULE DISCOVERY DEMONSTRATION")
    print("=" * 60)
    
    # Test different datasets
    datasets_to_test = ['celeba', 'cub', 'awa', 'chexpert']
    
    for dataset_name in datasets_to_test:
        print(f"\n{'='*60}")
        print(f"TESTING {dataset_name.upper()} DATASET")
        print(f"{'='*60}")
        
        # Create discovery system
        discovery = UniversalCBMDiscovery(dataset_name)
        
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 5000
        n_concepts = len(discovery.config.concept_names)
        
        # Generate realistic concept data based on domain
        if dataset_name == 'celeba':
            concept_data = generate_celeba_like_data(n_samples, discovery.config.concept_names)
        elif dataset_name == 'cub':
            concept_data = generate_cub_like_data(n_samples, discovery.config.concept_names)
        elif dataset_name == 'awa':
            concept_data = generate_awa_like_data(n_samples, discovery.config.concept_names)
        else:
            concept_data = np.random.rand(n_samples, n_concepts)
        
        # Discover rules
        groups, constraints = discovery.discover_all_rules(concept_data)
        
        # Show sample results
        print(f"\nSAMPLE RESULTS:")
        print(f"Concept groups: {list(groups.keys())}")
        print(f"Top constraints:")
        for i, constraint in enumerate(constraints[:3]):
            print(f"  {i+1}. {constraint}")

def generate_celeba_like_data(n_samples: int, concept_names: List[str]) -> np.ndarray:
    """Generate realistic CelebA-like concept data"""
    np.random.seed(42)
    data = np.random.rand(n_samples, len(concept_names))
    
    # Add realistic correlations for CelebA
    for i in range(n_samples):
        # Hair color mutual exclusion
        hair_colors = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']
        hair_indices = [concept_names.index(c) for c in hair_colors if c in concept_names]
        if hair_indices:
            dominant = np.random.choice(hair_indices)
            for idx in hair_indices:
                if idx == dominant:
                    data[i, idx] = np.random.beta(4, 1)
                else:
                    data[i, idx] = np.random.beta(1, 5)
        
        # Gender implications
        if 'Male' in concept_names and 'Mustache' in concept_names:
            male_idx = concept_names.index('Male')
            mustache_idx = concept_names.index('Mustache')
            
            if data[i, male_idx] > 0.7:
                data[i, mustache_idx] = np.random.beta(2, 3)
            else:
                data[i, mustache_idx] = np.random.beta(1, 8)
    
    return data

def generate_cub_like_data(n_samples: int, concept_names: List[str]) -> np.ndarray:
    """Generate realistic CUB-like concept data"""
    np.random.seed(42)
    data = np.random.rand(n_samples, len(concept_names))
    
    # Add bird-specific correlations
    for i in range(n_samples):
        # Bill shape implications
        if 'bill_shape_hooked' in concept_names and 'shape_hawks_like' in concept_names:
            hooked_idx = concept_names.index('bill_shape_hooked')
            hawks_idx = concept_names.index('shape_hawks_like')
            
            if data[i, hooked_idx] > 0.8:
                data[i, hawks_idx] = np.random.beta(4, 1)
    
    return data

def generate_awa_like_data(n_samples: int, concept_names: List[str]) -> np.ndarray:
    """Generate realistic AwA-like concept data"""
    np.random.seed(42)
    data = np.random.rand(n_samples, len(concept_names))
    
    # Add animal-specific correlations
    for i in range(n_samples):
        # Water animals can swim
        if 'water' in concept_names and 'swim' in concept_names:
            water_idx = concept_names.index('water')
            swim_idx = concept_names.index('swim')
            
            if data[i, water_idx] > 0.7:
                data[i, swim_idx] = np.random.beta(4, 1)
    
    return data

if __name__ == "__main__":
    demonstrate_universal_discovery()