from openai import OpenAI
import pandas as pd
import json
from typing import List
import time
import re
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key= os.getenv("OPEN_API_KEY"),
    base_url= os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/")
    # print(api_key)
)

PROMPT_TEMPLATE = """You are an expert in parsing product names from a vaping, CBD, kratom, and related products dataset. For each product name provided, extract the following attributes:
- Name: The full product name as provided.
- Brand: The brand name (e.g., "EBDESIGN", "Fruit Monster"). If multiple words form the brand, include them (e.g., "Fruit Monster").
- Model: The specific model or product code (e.g., "BC5000", "Go Stik"). Only extract if explicitly mentioned.
- Nicotine%: The nicotine strength as a decimal (e.g., 5% as 0.05). Only extract if explicitly stated.
- Flavor: The flavor, if explicitly mentioned (e.g., "Strawberry Cheesecake"). Do not infer flavors unless clearly stated.
- PackDetails: Details about pack size or volume (e.g., "5PK", "100ML", "10 Pack"). Extract exact terms when possible.
- PuffCount: The puff count for disposables (e.g., "15000" for "15000 Puffs"). Only extract if explicitly mentioned.

If an attribute is not explicitly mentioned or cannot be inferred with high confidence, return null for that attribute. Return the output as a JSON array of objects, where each object contains the attributes for one product. Handle inconsistencies in naming conventions and prioritize accuracy.

Example Inputs and Outputs:

Input: "EBDESIGN BC5000 DISPOSABLE 5% Nicotine"
Output: ```json
{{
  "Name": "EBDESIGN BC5000 DISPOSABLE 5% Nicotine",
  "Brand": "EBDESIGN",
  "Model": "BC5000",
  "Nicotine%": 0.05,
  "Flavor": null,
  "PackDetails": null,
  "PuffCount": null
}}
```

Input: "Fruit Monster PREMIUM ELIQUID 100ML"
Output: ```json
{{
  "Name": "Fruit Monster PREMIUM ELIQUID 100ML",
  "Brand": "Fruit Monster",
  "Model": null,
  "Nicotine%": null,
  "Flavor": null,
  "PackDetails": "100ML",
  "PuffCount": null
}}
```

Input: "BUDGET D8 1 GRAM CARTRIDGES STRAWBERRY CHEESECAKE"
Output: ```json
{{
  "Name": "BUDGET D8 1 GRAM CARTRIDGES STRAWBERRY CHEESECAKE",
  "Brand": "BUDGET D8",
  "Model": null,
  "Nicotine%": null,
  "Flavor": "Strawberry Cheesecake",
  "PackDetails": "1 GRAM",
  "PuffCount": null
}}
```

Input: "CBD KRATOM VAPES DETOX"
Output: ```json
{{
  "Name": "CBD KRATOM VAPES DETOX",
  "Brand": "CBD KRATOM",
  "Model": null,
  "Nicotine%": null,
  "Flavor": null,
  "PackDetails": "DETOX",
  "PuffCount": null
}}
```

Input: "Chris Brown CB15K Puffs Disposable 5PK"
Output: ```json
{{
  "Name": "Chris Brown CB15K Puffs Disposable 5PK",
  "Brand": "Chris Brown",
  "Model": "CB15K",
  "Nicotine%": null,
  "Flavor": null,
  "PackDetails": "5PK",
  "PuffCount": "15000"
}}
```

Now process the following product names and return a JSON array of objects with the attributes for each:
{product_list}
"""

def extract_attributes_batch(products: List[str], batch_size: int = 50) -> List[dict]:
    """
    Extract attributes from a list of product names using OpenAI's API in batches.
    
    Args:
        products: List of product names.
        batch_size: Number of products to process per API call.
    
    Returns:
        List of dictionaries containing extracted attributes.
    """
    results = []
    
    for i in range(0, len(products), batch_size):
        batch = products[i:i + batch_size]
        
        # Format the batch as a numbered list
        product_list = "\n".join([f"{j+1}. {p}" for j, p in enumerate(batch)])
        prompt = PROMPT_TEMPLATE.format(product_list=product_list)
        
        try:
            # Call OpenAI API
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.0
            )
            
            # Parse the response
            response_text = response.choices[0].message.content
            response_text = re.sub(r'^```json\n|\n```$', '', response_text.strip())
            batch_results = json.loads(response_text)
            results.extend(batch_results)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {str(e)}")
            
            results.extend([{
                "Name": p,
                "Brand": None,
                "Model": None,
                "Nicotine%": None,
                "Flavor": None,
                "PackDetails": None,
                "PuffCount": None
            } for p in batch])
        
        # Avoid hitting rate limits
        time.sleep(1)
    
    return results

def main():
    # Sample DataFrame for demonstration
    df = pd.read_csv("Combined_LandingPage_Scrapped_Products_DallasWholesalers_OtherRetailers.csv")
    
    # Extract unique products to reduce API calls
    unique_products = df["Products"].unique().tolist()
    print(f"Processing {len(unique_products)} unique products...")
    
    # Extract attributes
    extracted_data = extract_attributes_batch(unique_products, batch_size=50)
    
    # Convert to DataFrame
    extracted_df = pd.DataFrame(extracted_data)
    
    # Merge with original DataFrame to restore duplicates
    new_df = df.merge(extracted_df, left_on="Products", right_on="Name", how="left")
    
    # Save to CSV
    output_file = "extracted_products.csv"
    new_df[["Name", "Brand", "Model", "Nicotine%", "Flavor", "PackDetails", "PuffCount"]].to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print sample output
    print("\nSample of extracted data:")
    print(new_df[["Name", "Brand", "Model", "Nicotine%", "Flavor", "PackDetails", "PuffCount"]].head())

if __name__ == "__main__":
    main()