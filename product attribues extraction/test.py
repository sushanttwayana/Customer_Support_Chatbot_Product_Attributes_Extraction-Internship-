import pandas as pd
import os
from kor import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain_openai import ChatOpenAI  # Updated import
from datetime import datetime
import json
import re
import time
from typing import Dict, Optional, List

# Initialize LangChain's OpenAI wrapper with better error handling
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.0,
    max_tokens=2000,
    request_timeout=60,  # Add timeout
    max_retries=3  # Add retries
)

# schema for the llm with specific instrutions
vaping_product_schema = Object(
    id="product",
    description=(
        "Extract detailed information about vaping and smoking products from product names. "
        "Focus on accurately identifying the brand first, then other attributes. "
        "For ambiguous cases, make reasonable inferences based on common industry terms."
        
    ),
    attributes=[
        Text(
            id="brand",
            description=(
                "The manufacturer or brand name. Look for trademarked brand names at the beginning of product names. "
                "If you cannot find the brand name from the products leave it empty or null please donot force yourself to create the brand name that might not exist in real."
                "Examples: 'Geek Bar', 'Lost Mary', 'NANO SMOKE', 'Juicy Bar', 'Rookie Bar', 'Mya'"
            ),
            examples=[
                ("Juicy Bar JB5000 Disposable", "Juicy Bar"),
                ("Mya Oro MX Hookah", "Mya"),
                ("NANO SMOKE HORECA LED HOOKAH", "NANO SMOKE"),
                ("20000 PUFFS VABEEN CYBERFLEX DISPOSABLE", "Vabeen"),

            ],
            many=False,
        ),
        Text(
            id="model",
            description=(
                "The specific model name, number, or product code. "
                "Typically includes numbers or specific model names. "
                "Examples: 'JB5000', 'CYBERFLEX', 'PRISM', 'MX 1A5'"
            ),
            examples=[
                ("Juicy Bar JB5000 Disposable", "JB5000"),
                ("20000 PUFFS VABEEN CYBERFLEX DISPOSABLE", "CYBERFLEX"),
                ("23\" MYA ORO MX 1A5 HOOKAH", "MX 1A5"),
                ("NANO SMOKE HORECA LED HOOKAH", "HORECA"),
                ("20000 PUFFS VABEEN CYBERFLEX DISPOSABLE", "Cyberflex")
            ],
            many=False,
        ),
        Text(
            id="flavor",
            description=(
                "The product flavor if mentioned. Look for descriptive terms at the end of product names. "
                "Examples: 'Blue Raspberry', 'Watermelon Ice', 'Mint'"
            ),
            examples=[
                ("Geek Bar Pulse X Blue Raspberry", "Blue Raspberry"),
                ("Lost Mary MO20000 Watermelon Ice", "Watermelon Ice"),
            ],
            many=False,
        ),
        Text(
            id="category",
            description=(
                "The product category. Choose from: "
                "'Disposable Vape', 'E-Liquid', 'Hookah', 'Water Pipe', 'Kratom', "
                "'CBD', 'Cigar', 'Battery', 'Accessories', 'Dab Rig', 'Other'. "
                "Determine based on product type and keywords."
            ),
            examples=[
                ("10\" NANO SMOKE HORECA LED HOOKAH", "Hookah"),
                ("7\" EVRST ART COLOR DAB RIG", "Dab Rig"),
                ("5000 PUFFS JUICY BAR JB5000 DISPOSABLE", "Disposable Vape"),
                ("380MAH VARIABLE VOLTAGE BATTERY", "Battery"),
                ("20000 PUFFS VABEEN CYBERFLEX DISPOSABLE", "Disposable Vape")
            ],
            many=False,
        ),
        # Number(
        #     id="puff_count",
        #     description=(
        #         "The number of puffs if specified in the product name. "
        #         "Extract only the numeric value."
        #     ),
        #     examples=[
        #         ("20000 PUFFS VABEEN CYBERFLEX", 20000),
        #         ("25000 PUFFS ROOKIE BAR", 25000),
        #     ],
        #     many=False,
        # )
    ],
    examples=[
        (
            "Geek Bar Pulse X 25K Puffs Disposable Blue Raspberry (Box of 5)",
            [
                {
                    "brand": "Geek Bar",
                    "model": "Pulse X 25K",
                    "flavor": "Blue Raspberry",
                    "category": "Disposable Vape"
                }
            ]
        ),
        (
            "EBDESIGN BC5000 DISPOSABLE 5% Nicotine",
            [
                {
                    "brand": "EBDESIGN",
                    "model": "BC5000",
                    "flavor": "Disposable Vape",
                    "category": "Disposable Vape"
                }
            ]
        ),
        (
            "Fruit Monster PREMIUM ELIQUID 100ML",
            [
                {
                    "brand": "Fruit Monster",
                    "model": "PREMIUM ELIQUID",
                    "flavor": "",
                    "category": "Vape Juince (ELIQUID)"
                }
            ]
        ),
        (
            "CBD KRATOM VAPES DETOX",
            [
                {
                    "brand": "CBD KRATOM",
                    "model": "",
                    "flavor": "",
                    "category": "Vape"
                }
            ]
        ),

        (
            "Chris Brown CB15K Puffs Disposable 5PK",
            [
                {
                    "brand": "Chris Brown",
                    "model": "CB15K",
                    "flavor": "",
                    "category":"Disposable Vape"
                }
            ]
        ),
        (
            "BUDGET D8 1 GRAM CARTRIDGES STRAWBERRY CHEESECAKE",
            [
                {
                    "brand": "Delta-8 THC",
                    "model": "CB15K",
                    "flavor": "STRAWBERRY CHEESECAKE",
                    "category":"Vapes"
                }
            ]
        ),
        (
            "Lost Mary MO 20000 Pro Disposable Vape Watermelon Ice (Box of 5)",
            [
                {
                    "brand": "Lost Mary",
                    "model": "MO 20000 Pro",
                    "flavor": "Watermelon Ice",
                    "category":"Disposable Vape"
                }
            ]
        ),
        (
            "Fruit Monster PREMIUM ELIQUID 100ML",
            [
                {
                    "brand": "Monster Vape Labs",
                    "model": "PREMIUM ELIQUID 100ML",
                    "flavor": "",
                    "category":"Vape Juice"
                }
            ]
        ),
        (
            "KadoBar NI40000 Ice + Nicotine Control Disposable Vape",
            [
                {
                    "brand": "KadoBar",
                    "model": "NI40000",
                    "flavor": "Ice",
                    "category":"Disposable Vape Devices"
                }
            ]
        ),
        (
            "7\" EVRST ART COLOR DAB RIG WATER PIPE WITH 14MM MALE BANGER",
            {
                "brand": "EVRST",
                "model": "ART COLOR",
                "flavor": "",
                "category": "Dab Rig",
            },
        ),
        (
            "20000 PUFFS VABEEN CYBERFLEX DISPOSABLE",
            {
                "brand": "VABEEN",
                "model": "CYBERFLEX",
                "flavor": "",
                "category": "Disposable Vape",
            },
        ),

        (
            "10\" NANO SMOKE HORECA LED HOOKAH",
            {
                "brand": "NANO SMOKE",
                "model": "HORECA LED",
                "flavor": "",
                "category": "Hookah",
            },
        ),
        (
            "12″ Painted Zig Zag Single Perc Water Pipe",
            {
                "brand": "",
                "model": "Painted Zig Zag Single Perc",
                "flavor": "",
                "category": "Water Pipe",
            },
        ),
        (
            "5000 PUFFS JUICY BAR JB5000 DISPOSABLE",
            {
                "brand": "Juicy Bar",
                "model": "JB5000",
                "flavor": "",
                "category": " Disposable Vape",
            },
        ),
        (
            "14 PAINTED TRIPLE DOME PERC WATER PIPE - ASSORTED",
            {
                "brand": "",
                "model": "TRIPLE DOME PERC",
                "flavor": "ASSORTED",
                "category": "Water Pipe",
            },
        ),

    ],
    many=False,
)

# Brand dictionary for fallback identification
KNOWN_BRANDS = {
    'NANO SMOKE': ['NANO SMOKE', 'NANOSMOKE'],
    'VABEEN': ['VABEEN'],
    'MYA': ['MYA'],
    'Geek Bar': ['GEEK BAR', 'GEEKBAR'],
    'Lost Mary': ['LOST MARY', 'LOSTMARY'],
    'Juicy Bar': ['JUICY BAR', 'JUICYBAR'],
    'Rookie Bar': ['ROOKIE BAR', 'ROOKIEBAR'],
    'SPACEMAN': ['SPACEMAN'],
    'XTRON': ['XTRON'],
    'KFBAR': ['KFBAR', 'KF BAR'],
    'EVRST': ['EVRST', 'EVEREST'],
    '3 CHI': ['3 CHI', '3CHI', 'CHI'],
    'Chris Brown': ['CHRIS BROWN'],
    'KadoBar': ['KADOBAR', 'KADO BAR'],
    'EBDESIGN': ['EBDESIGN', 'EB DESIGN'],
    'Fruit Monster': ['FRUIT MONSTER'],
    'Monster Vape Labs': ['MONSTER VAPE', 'MONSTER'],
    'BUDGET': ['BUDGET'],
    'YOCAN': ['YOCAN'],
    'OXBAR': ['OXBAR'],
    'PULSAR': ['PULSAR'],
    'PILLOWZ': ['PILLOWZ'],
    'STIIIZY': ['STIIIZY'],
    'ALOHA SUN': ['ALOHA SUN'],
    'SKWEZED': ['SKWEZED'],
    'YAMI VAPORS': ['YAMI VAPORS'],
    'SUGOI VAPOR': ['SUGOI VAPOR'],
    'ROAD TRIP': ['ROAD TRIP'],
    'MAGIC MAZE': ['MAGIC MAZE'],
    'LUCY': ['LUCY'],
    'Orion Bar': ['ORION BAR', 'ORIONBAR'],
    'Air Factory': ['AIR FACTORY'],
    'Coastal Cloud': ['COASTAL CLOUD'],
    'Vapetasia': ['VAPETASIA'],
    'Pod Juice': ['POD JUICE'],
    'Esco Bars': ['ESCO BARS', 'ESCO'],
    'Chris Brown': ['CHRIS BROWN'],
    'Aloha Sun': ['ALOHA SUN'],
    'Dutch Masters': ['DUTCH MASTERS'],
    'Buddys Hemp': ['BUDGET', 'BUDGET D8']
}

KNOWN_MODELS = {
    # Disposable Vapes
    'BC5000': ['BC5000', 'BC 5000'],
    'Pulse X': ['PULSE X', 'PULSE-X', 'PULSE X 25K'],
    'Sky View': ['SKY VIEW', 'SKYVIEW'],
    'JB5000': ['JB5000', 'JB 5000'],
    'JB7500 Pro': ['JB7500 PRO', 'JB 7500 PRO'],
    'MX25000': ['MX25000', 'MX 25000'],
    'DC25000': ['DC25000', 'DC 25000'],
    'Rookie 25000': ['ROOKIE 25000', 'ROOKIE BAR 25000'],
    '10K Pro': ['10K PRO', '10KPRO'],
    'X 10000': ['X 10000', 'X10000'],
    'XTRON 30000': ['XTRON 30000', 'XTRON30000'],
    
    # Hookah Models
    'ORO MX 1A5': ['ORO MX 1A5', 'MX 1A5'],
    'HORECA LED': ['HORECA LED', 'HORECA'],
    'ACID': ['ACID'],
    
    # Special Cases
    'CB15K': ['CB15K', 'CB 15K'],  # Chris Brown model
    'Cyberflex': ['CYBERFLEX'],
    'Prism': ['PRISM'],
    
    # Number-Based Models
    '40000': ['40000', '40K', '40K PUFFS'],
    '25000': ['25000', '25K', '25K PUFFS'],
    '20000': ['20000', '20K', '20K PUFFS'],
    '15000': ['15000', '15K', '15K PUFFS'],
    '10000': ['10000', '10K', '10K PUFFS'],
    '5000': ['5000', '5K', '5K PUFFS']
}

def extract_brand_conservative(product_name: str) -> str:
    """Conservative brand extraction - only return if very confident."""
    name_upper = product_name.upper()
    
    # Check against known brands only
    for brand, variations in KNOWN_BRANDS.items():
        for variation in variations:
            if variation in name_upper:
                # Additional validation - brand should be at the beginning or clearly separated
                words = name_upper.split()
                if any(word.startswith(variation) or variation in word for word in words[:3]):
                    return brand
    
    return ""  # Return empty if not confident

def create_kor_extraction_chain():
    """Create the Kor extraction chain with better error handling."""
    try:
        chain = create_extraction_chain(
            node=vaping_product_schema,
            llm=llm,
            encoder_or_encoder_class="json",
        )
        print("Chain created successfully")
        return chain
    except Exception as e:
        print(f"Chain creation failed: {str(e)}")
        raise e

def extract_product_info_with_retry(product_name: str, chain, max_retries: int = 3) -> Dict:
    """Extract information with retry logic and better error handling."""
    
    for attempt in range(max_retries):
        try:
            # Add small delay between retries
            if attempt > 0:
                time.sleep(1)
            
            response = chain.invoke({"text": product_name})
            
            # Parse response
            product_data = {}
            
            if isinstance(response, dict):
                # Try different response structures
                if 'text' in response and isinstance(response['text'], dict):
                    if 'data' in response['text'] and 'product' in response['text']['data']:
                        product_data = response['text']['data']['product']
                    elif 'product' in response['text']:
                        product_data = response['text']['product']
                elif 'data' in response and isinstance(response['data'], dict):
                    if 'product' in response['data']:
                        product_data = response['data']['product']
                elif 'product' in response:
                    product_data = response['product']
            
            # Handle list responses
            if isinstance(product_data, list) and len(product_data) > 0:
                product_data = product_data[0]
            
            # Ensure we have a dict
            if not isinstance(product_data, dict):
                product_data = {}
            
            # Extract and clean fields
            brand = str(product_data.get('brand', '')).strip()
            model = str(product_data.get('model', '')).strip()
            flavor = str(product_data.get('flavor', '')).strip()
            category = str(product_data.get('category', '')).strip()
            
            # Validate brand - be conservative
            if brand and brand.lower() in ['none', 'n/a', 'unknown', 'generic']:
                brand = ""
            
            # If LLM didn't find brand, try conservative fallback
            if not brand:
                brand = extract_brand_conservative(product_name)
            
            # Set default category if empty
            if not category or category.lower() in ['other', 'unknown']:
                category = categorize_product(product_name)
            
            return {
                'brand': brand,
                'model': model,
                'flavor': flavor,
                'category': category,
                'extraction_method': 'LLM'
            }
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed for '{product_name}': {str(e)}")
            if attempt == max_retries - 1:
                # Final fallback to rule-based extraction
                return extract_fallback(product_name)
    
    return extract_fallback(product_name)

def categorize_product(product_name: str) -> str:
    """Categorize product based on keywords."""
    name_upper = product_name.upper()
    
    if any(keyword in name_upper for keyword in ['ESCO BARS', 'MESH', 'DISPOSABLE', 'PUFF']):
        return 'Disposable Vape'
    elif any(keyword in name_upper for keyword in ['FRUITIA', 'E-LIQUID', 'ELIQUID']):
        return 'E-Liquid'
    elif any(keyword in name_upper for keyword in ['BACKWOOD', 'SWISHER', 'CIGAR', 'TOBACCO']):
        if 'TOBACCO' in name_upper and 'BACKWOOD' not in name_upper:
            return 'Tobacco'
        return 'Cigar'
    elif any(keyword in name_upper for keyword in ['SALT', 'SODA', 'JUICE', 'HERSHEY', 'SNICKER', 'HARIBO']):
        return 'Food & Beverage'
    elif any(keyword in name_upper for keyword in ['NAPKIN', 'PAPER']):
        return 'Paper Products'
    elif any(keyword in name_upper for keyword in ['PLASTIC', 'CONTAINER', 'CUTLERY']):
        return 'Plastic Products'
    else:
        return 'Other'

def extract_fallback(product_name: str) -> Dict:
    """Complete fallback extraction using only rules."""
    return {
        'brand': extract_brand_conservative(product_name),
        'model': '',
        'flavor': '',
        'category': categorize_product(product_name),
        'extraction_method': 'Fallback'
    }

def extract_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Process dataframe and extract attributes with better error handling."""
    print("Initializing extraction chain...")
    
    try:
        chain = create_kor_extraction_chain()
    except Exception as e:
        print(f"Failed to create extraction chain: {str(e)}")
        print("Using fallback rule-based extraction only...")
        return extract_all_fallback(df)
    
    results = []
    total_products = len(df)
    print(f"Starting extraction for {total_products} products...")
    
    successful_extractions = 0
    failed_extractions = 0
    
    for i, row in df.iterrows():
        product_name = str(row['Products']).strip()
        
        if i % 10 == 0:  # Progress update every 10 items
            print(f"Processing item {i+1}/{total_products}: {product_name[:50]}...")
        
        extracted = extract_product_info_with_retry(product_name, chain)
        
        if extracted['extraction_method'] == 'LLM':
            successful_extractions += 1
        else:
            failed_extractions += 1
        
        results.append({
            'Products': product_name,
            'brand': extracted['brand'],
            'model': extracted['model'],
            'flavor': extracted['flavor'],
            'category': extracted['category'],
            'extraction_method': extracted['extraction_method']
        })
    
    print(f"\nExtraction completed:")
    print(f"Successful LLM extractions: {successful_extractions}")
    print(f"Fallback extractions: {failed_extractions}")
    
    return pd.DataFrame(results)

def extract_all_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback extraction for all products."""
    print("Using rule-based extraction for all products...")
    results = []
    
    for i, row in df.iterrows():
        product_name = str(row['Products']).strip()
        extracted = extract_fallback(product_name)
        
        results.append({
            'Products': product_name,
            'brand': extracted['brand'],
            'model': extracted['model'],
            'flavor': extracted['flavor'],
            'category': extracted['category'],
            'extraction_method': extracted['extraction_method']
        })
    
    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    try:
        # Read input file
        input_file = 'final_preprocessed.csv'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        df = pd.read_csv(input_file)
        print(f"Loaded {len(df)} products from {input_file}")
        
        # Verify required columns
        if 'Products' not in df.columns:
            raise ValueError(f"'Products' column not found. Available columns: {df.columns.tolist()}")
        
        # Process extraction
        test_mode = True
        if test_mode:
            print("\nRunning in TEST MODE (first 70 products only)")
            df = df.head(70)  
    
        print("Starting attribute extraction...")
        result_df = extract_attributes(df)
        
        # Save results
        output_file = f"extracted_attributes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Display sample results
        print("\nSample extraction results:")
        for i in range(min(10, len(result_df))):
            row = result_df.iloc[i]
            print(f"\nProduct: {row['Products']}")
            print(f"  Brand: '{row['brand']}' | Model: '{row['model']}' | Category: '{row['category']}' | Method: {row['extraction_method']}")
        
        # Summary statistics
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total products processed: {len(result_df)}")
        
        # Count non-empty brands
        non_empty_brands = len(result_df[result_df['brand'].str.strip() != ''])
        print(f"Products with brands identified: {non_empty_brands} ({non_empty_brands/len(result_df)*100:.1f}%)")
        
        non_empty_models = len(result_df[result_df['model'].str.strip() != ''])
        print(f"Products with models identified: {non_empty_models}")
        
        non_other_categories = len(result_df[result_df['category'] != 'Other'])
        print(f"Products with specific categories: {non_other_categories}")
        
        # Method breakdown
        method_counts = result_df['extraction_method'].value_counts()
        print(f"\nExtraction method breakdown:")
        for method, count in method_counts.items():
            print(f"  {method}: {count}")
        
        # Brand breakdown (only non-empty)
        brand_counts = result_df[result_df['brand'].str.strip() != '']['brand'].value_counts()
        if len(brand_counts) > 0:
            print(f"\nBrand breakdown:")
            for brand, count in brand_counts.head(10).items():  # Top 10 brands
                print(f"  {brand}: {count}")
        
        # Category breakdown
        category_counts = result_df['category'].value_counts()
        print(f"\nCategory breakdown:")
        for category, count in category_counts.items():
            print(f"  {category}: {count}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        import traceback
        traceback.print_exc()