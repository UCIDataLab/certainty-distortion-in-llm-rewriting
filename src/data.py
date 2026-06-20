
import datasets
import random
import os
import pandas as pd

from jinja2 import Template
from typing import List, Tuple
from utils import assign_uuid


# Hard coded
FORMATTED_INPUT_COL = "prompt"
UUID_COL = "uuid"


def load_dataset(
    input_path: str | List[str],
    jinja_template: str,
    seed=18274,
    uuid_col: str = None,
    id_cols: str | List[str] = None,
    shuffle_cols: List[dict] = None,
    subset: int = None,
    **kwargs) -> datasets.Dataset:
    # Step 1. Load dataset
    
    if os.path.isfile(input_path):
        df = pd.read_csv(input_path)
        dataset = datasets.Dataset.from_pandas(df)
    else:
        
        input_path = [input_path] if isinstance(input_path, str) else input_path
        dataset = datasets.load_dataset(*input_path, **kwargs)
    # [optional] Shuffle columns
    if shuffle_cols:
        rand = random.Random(seed)
        prefix = shuffle_cols.get("prefix", "prefix_")
        cols = shuffle_cols["cols"]
        
        def apply_shuffle_cols_(example, rand, prefix, cols, label_col):
            vals = [example[c] for c in cols if example[c]]
            rand.shuffle(vals)

            for i, v in enumerate(vals):
                example[f"{prefix}{i}"] = v
                if v == example[label_col]:
                    example[label_col] = chr(ord('A') + i)
            return example
            
        dataset = dataset.map(apply_shuffle_cols_, fn_kwargs={"cols": cols, "rand": rand, "prefix": prefix, "label_col": label_col}) # FIXME
        
    # Step 2. Apply the prompt
    template = Template(jinja_template)
    def _apply_prompt(example):
        example[FORMATTED_INPUT_COL] = template.render(**example)
        return example
    dataset = dataset.map(_apply_prompt)
    
    # Step 3. Create unique identifiers for each example
    if uuid_col is None:
        uuid_col = UUID_COL
    
    if uuid_col not in dataset.features:
        if id_cols is None:
            id_cols = list(dataset.features)
            
        def _get_uuid(example):
            example[uuid_col] = assign_uuid(example, cols=id_cols)
            return example
        dataset = dataset.map(_get_uuid)
    
    # Step 4. Add some columns for templated access later on
    if UUID_COL not in dataset.features: 
        dataset = dataset.add_column(UUID_COL, dataset[uuid_col])
         
    # Step 5. Convert to pandas dataframe
    df = dataset.to_pandas()
    if subset:
        df = df.head(subset)   
    return df
