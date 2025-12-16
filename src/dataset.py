import numpy as np
import polars as pl
from pathlib import Path
from torch.utils.data import Dataset

allowed_classes = [
    'SPB', 'GDOR', 'SXPHE', 'BCEP', 'DSCT', # Main-sequence Kappa pulsators
    'DCEP', 'ACEP', 'T2CEP', 'RVTAU', # 'CW', 'BLHER', # Classical Cepheids
    'EA', 'EB', 'EW', 'ELL', # Eclipsing binaries
    'LSP', 'M', 'SARG', 'SRA', 'SRB', 'SRC', 'SRD', 'SRS', # LPVs
    'YSO', #Young Stellar Objects
    'SDB', # Subdwarf B-stars
    'WD', # White dwarfs
    'RRAB', 'RRC', 'RRD', 'ARRD', # RR Lyrae stars 
    'BLUEVARS',
    'CP', # Chemically peculiar stars
    'SOLARLIKE', # Rotationally modulated and flaring solar-like stars
    'MICROLENSING',
    'AGN', # Active galactic nuclei
    'CST', # Constant stars
    'EP', # Exoplanet
    'OTHER',
] 

class_merger_mapper = {
    'BLHer': 'BLHER',
    'RVTau': 'RVTAU', 'RV': 'RVTAU',
    'TTS': 'YSO', 'CTTS': 'YSO', 'WTTS': 'YSO', 'IMTTS': 'YSO', 'GTTS': 'YSO', 'UXOR': 'YSO', 'PULS_PMS': 'YSO',    
    'V1093HER': 'SDB', 'V361HYA': 'SDB', 'sdB': 'SDB',
    'GWVIR': 'WD', 'ZZA': 'WD', 'ZZ': 'WD', 'EHM_ZZA': 'WD', 'V777HER': 'WD',
    'RRab': 'RRAB', 'RRab': 'RRAB',
    'RRc': 'RRC', 'RRd': 'RRD',
    'BE|GCAS|SDOR|WR': 'BLUEVARS',
    'BE': 'BLUEVARS', 'GCAS': 'BLUEVARS', 'SDOR': 'BLUEVARS', 'WR': 'BLUEVARS',
    'ACV|ROAM|ROAP|SXARI': 'CP', 'ACV|CP|MCP|ROAM|ROAP|SXARI': 'CP', 'ACV|roAm|roAp|SXARI': 'CP', 'ACV|roAm|roAp|SXARI': 'CP',
    'ACV': 'CP', 'MCP': 'CP', 'SXARI': 'CP', 'ROAP': 'CP', 
    'BY': 'SOLARLIKE', 'FLARES': 'SOLARLIKE', 'ROT': 'SOLARLIKE', 'SOLAR_LIKE': 'SOLARLIKE', 'RS': 'SOLARLIKE',
    'QSO': 'AGN', 'BLLAC': 'AGN', 'GALAXY': 'AGN',  'BLAZAR': 'AGN', 'BLAP': 'AGN',
    'ACYG': 'OTHER', 'CV': 'OTHER', 'HB': 'OTHER', 'HMXB': 'OTHER', 'PCEB': 'OTHER', 
    'SYST': 'OTHER', 'ZAND': 'OTHER', 'UV': 'OTHER', 'RCB': 'OTHER', 'SN': 'OTHER',
    'OSARG': 'SARG',
}

# PCEB, CV, SYST, ZAND Could be it own CV category?

def build_spurious_frequency_mask(left_bound: list[float], right_bound: list[float]) -> pl.Expr:
    assert len(left_bound) == len(right_bound), "Left and right bounds arrays have to be of same length"
    mask_exprs = [pl.col('frequency').log10().is_between(l, r, closed='both') for l, r in zip(left_bound, right_bound)]
    return pl.any_horizontal(mask_exprs)

#def spurious_frequencies_mask():
#    peak_left =  [-2.273, -1.987, -1.736, -1.678, -1.505, -1.435, -1.405, -1.379, -1.278, -1.237, -1.148, 0.602, 1.053, 1.187, 1.373]
#    peak_right = [-2.251, -1.975, -1.725, -1.667, -1.490, -1.424, -1.396, -1.368, -1.271, -1.228, -1.145, 0.603, 1.113, 1.214, 1.385]
#    mask_exprs = [pl.col('frequency').log10().is_between(l, r, closed='both') for l, r in zip(peak_left, peak_right)]
#   return pl.any_horizontal(mask_exprs)

def faint_spurious_frequency_mask():
    peak_left =  np.array([-2.290, -2.273, -2.238, -1.988, -1.970, -1.890, -1.740, -1.678, -1.505, -1.435, -1.405, -1.375, -1.303, -1.278, -1.237, -1.199, -1.148, 0.600, 1.053, 1.187, 1.371])
    peak_right = np.array([-2.280, -2.251, -2.228, -1.974, -1.955, -1.874, -1.720, -1.667, -1.490, -1.424, -1.396, -1.368, -1.296, -1.271, -1.228, -1.193, -1.145, 0.604, 1.113, 1.214, 1.390])
    return build_spurious_frequency_mask(peak_left.tolist(), peak_right.tolist())

def bright_spurious_frequency_mask():
    peak_left =  np.array([-2.273, -1.987, -1.888, -1.736, -1.678, -1.535, -1.505, -1.465, -1.435, -1.405, -1.379, -1.278, -1.257, -1.237, -1.199, -1.148, 0.602, 1.053, 1.187, 1.373])
    peak_right = np.array([-2.251, -1.975, -1.874, -1.725, -1.667, -1.531, -1.490, -1.462, -1.424, -1.396, -1.368, -1.271, -1.253, -1.228, -1.193, -1.145, 0.603, 1.113, 1.214, 1.385])
    return build_spurious_frequency_mask(peak_left.tolist(), peak_right.tolist())
        

def prepare_dataset(training_set_path: Path | str, 
                    only_taxonomy_classes: bool = False, 
                    only_sources_with_reliable_labels: bool = False,
                    max_logfap: float | None = None, 
                    filter_spurious_frequencies: bool = False,
                    min_sources_per_class: int | None = None, 
                    max_sources_per_class: int | None = None,
                    subsampling_seed: int = 1234
                   ):
    if isinstance(training_set_path, str):
        training_set_path = Path(training_set_path)
    if not training_set_path.exists():
        raise FileNotFoundError(f"Training set parquet not found at {training_set_path}")
    is_faint_dataset = 'FAINT' in training_set_path.name
    training_set = pl.scan_parquet(training_set_path)
    # The only difference between BRIGHT and FAINT taxonomies is in the semi regulars
    local_mapper = class_merger_mapper.copy()
    if is_faint_dataset:
        local_mapper = local_mapper | {k: 'SR' for k in ['SRA', 'SRB', 'SRC', 'SRD', 'SRS']}
    training_set = training_set.with_columns(
        pl.col('primary_type').replace(local_mapper).alias('label')
    )
    if only_taxonomy_classes:
        local_allowed = allowed_classes.copy()
        if is_faint_dataset:
            local_allowed = local_allowed + ['SR']
        training_set = training_set.filter(
            pl.col('label').is_in(local_allowed)
        )
    if only_sources_with_reliable_labels:
        df_reliable = pl.scan_parquet(training_set_path.parent / 'reliable_labels.parquet').select('sourceid')
        training_set = training_set.join(df_reliable, on='sourceid')    
    if max_logfap is not None:
        training_set = training_set.filter(pl.col('fap_log_10').le(max_logfap))
    if filter_spurious_frequencies:
        if is_faint_dataset:
            training_set = training_set.filter(~faint_spurious_frequency_mask())
        else:
            training_set = training_set.filter(~bright_spurious_frequency_mask())
        
    training_set = training_set.collect()
    if min_sources_per_class is not None:
        selected_classes = training_set.group_by('label').len().filter(
            pl.col('len') > min_sources_per_class
        )['label'].to_list()
        training_set = training_set.filter(
            pl.col('label').is_in(selected_classes)
        )
    if max_sources_per_class is not None:        
        training_set = training_set.group_by('label', maintain_order=True).map_groups(
            lambda g: g.sample(n=max_sources_per_class, seed=subsampling_seed) if g.height > max_sources_per_class else g
        )
    return training_set.sort('sourceid')