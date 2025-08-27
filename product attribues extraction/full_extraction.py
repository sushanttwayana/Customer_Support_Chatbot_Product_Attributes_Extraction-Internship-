import pandas as pd
import os
from kor import create_extraction_chain
from kor.nodes import Object, Text, Number
from langchain.chat_models import ChatOpenAI
from datetime import datetime
import json
import re

# Initialize LangChain's OpenAI wrapper
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.0,
    max_tokens=2000
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

def extract_brand_fallback(product_name: str) -> str:
    """Fallback brand extraction using known brands dictionary."""
    name_upper = product_name.upper()
    
    for brand, variations in KNOWN_BRANDS.items():
        for variation in variations:
            if variation in name_upper:
                return brand
    
    # # If no known brand found, try to extract first meaningful word
    # words = name_upper.split()
    # skip_words = {'10"', '12"', '14"', '20"', '23"', '25"', '30"', '7"', 
    #               '20000', '25000', '30000', '5000', 'PUFFS', 'PUFF', 
    #               'DISPOSABLE', 'THE', 'A', 'AN'}
    
    # for word in words:
    #     clean_word = word.replace('"', '').replace('″', '').replace('%', '')
    #     if clean_word not in skip_words and len(clean_word) > 2:
    #         return clean_word.title()
    
    # return ""

def create_kor_extraction_chain():
    """Create the Kor extraction chain with proper configuration."""
    try:
        # Try different chain creation methods
        methods = [
            lambda: create_extraction_chain(
                node=vaping_product_schema,
                llm=llm,
                encoder_or_encoder_class="json",
            ),
            lambda: create_extraction_chain(
                schema=vaping_product_schema,
                llm=llm,
                encoder_or_encoder_class="json",
            ),
            lambda: create_extraction_chain(
                node=vaping_product_schema,
                llm=llm,
            ),
            lambda: create_extraction_chain(vaping_product_schema, llm),
        ]
        
        for i, method in enumerate(methods, 1):
            try:
                chain = method()
                print(f"Chain created successfully using method {i}")
                return chain
            except Exception as e:
                print(f"Method {i} failed: {str(e)}")
                continue
        
        raise Exception("All chain creation methods failed")
        
    except Exception as e:
        print(f"Chain creation failed: {str(e)}")
        raise e

def extract_product_info(product_name: str, chain):
    """Extract information from a single product name with multiple fallbacks."""
    try:
        # Try different invocation methods
        response = None
        methods = [
            lambda: chain.invoke({"text": product_name}),
            lambda: chain.invoke(product_name),
            lambda: chain.run(product_name),
            lambda: chain.invoke({"input": product_name}),
        ]
        
        for method in methods:
            try:
                response = method()
                break
            except:
                continue
        
        if response is None:
            raise Exception("All invocation methods failed")
        
        # Parse response with multiple fallbacks
        product_data = {}
        
        if isinstance(response, dict):
            # Try different response structures
            possible_paths = [
                lambda r: r.get('text', {}).get('data', {}).get('product', {}),
                lambda r: r.get('data', {}).get('product', {}),
                lambda r: r.get('product', {}),
                lambda r: r.get('text', {}),
                lambda r: r,
            ]
            
            for path in possible_paths:
                try:
                    product_data = path(response)
                    if isinstance(product_data, dict) and product_data:
                        break
                except:
                    continue
        
        # Handle list responses
        if isinstance(product_data, list) and len(product_data) > 0:
            product_data = product_data[0]
        
        # Ensure we have a dict
        if not isinstance(product_data, dict):
            product_data = {}
        
        # Extract fields with fallbacks
        brand = product_data.get('brand', '').strip()
        model = product_data.get('model', '').strip()
        flavor = product_data.get('flavor', '').strip()
        category = product_data.get('category', 'Other').strip()
        puff_count = product_data.get('puff_count', '')
        
        # Brand fallback - this is critical
        if not brand:
            brand = extract_brand_fallback(product_name)
            print(f"Used fallback brand extraction for '{product_name}': '{brand}'")
        
        # Category fallback
        if category == 'Other' or not category:
            name_upper = product_name.upper()
            if 'DISPOSABLE' in name_upper or 'PUFF' in name_upper:
                category = 'Disposable Vape'
            elif 'HOOKAH' in name_upper:
                category = 'Hookah'
            elif 'WATER PIPE' in name_upper or 'PERC' in name_upper:
                category = 'Water Pipe'
            elif 'DAB RIG' in name_upper:
                category = 'Dab Rig'
            elif 'BATTERY' in name_upper or 'MAH' in name_upper:
                category = 'Battery'
            elif 'CIGARILLO' in name_upper:
                category = 'Cigar'
        
        # # Puff count fallback
        # if not puff_count:
        #     puff_match = re.search(r'(\d+)K?\s*PUFF', product_name.upper())
        #     if puff_match:
        #         puff_count = int(puff_match.group(1))
        #         if 'K' in puff_match.group(0):
        #             puff_count *= 1000
        
        return {
            'brand': brand,
            'model': model,
            'flavor': flavor,
            'category': category,
            'puff_count': puff_count if puff_count else '',
        }
        
    except Exception as e:
        print(f"Error processing '{product_name}': {str(e)}")
        # Complete fallback extraction
        return {
            'brand': extract_brand_fallback(product_name),
            'model': '',
            'flavor': '',
            'category': 'Other',
            'puff_count': '',
        }

def extract_attributes(df: pd.DataFrame, batch_size: int = 100) -> pd.DataFrame:
    """Process dataframe and extract attributes."""
    print("Initializing extraction chain...")
    
    try:
        chain = create_kor_extraction_chain()
    except Exception as e:
        print(f"Failed to create extraction chain: {str(e)}")
        print("Falling back to rule-based extraction...")
        return manual_extract_attributes(df)
    
    results = []
    total_products = len(df)
    print(f"Starting extraction for {total_products} products...")
    
    for i, row in df.iterrows():
        product_name = str(row['Products']).strip()
        if i % 5 == 0:  # More frequent progress updates
            print(f"Processing item {i+1}/{total_products}: {product_name[:50]}...")
        
        extracted = extract_product_info(product_name, chain)
        results.append({
            'Products': product_name,
            **extracted
        })
    
    return pd.DataFrame(results)

def manual_extract_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """Fallback manual extraction using rules and known brands."""
    print("Using manual extraction with enhanced brand detection...")
    results = []
    
    for i, row in df.iterrows():
        product_name = str(row['Products']).strip()
        name_upper = product_name.upper()
        
        # Extract brand using known brands dictionary
        brand = extract_brand_fallback(product_name)
        
        # Extract other attributes
        model = ''
        flavor = ''
        category = 'Other'
        puff_count = ''
        
        # Extract puff count
        puff_match = re.search(r'(\d+)K?\s*PUFF', name_upper)
        if puff_match:
            puff_count = int(puff_match.group(1))
            if 'K' in puff_match.group(0):
                puff_count *= 1000
        
        # Determine category
        if 'DISPOSABLE' in name_upper or 'PUFF' in name_upper:
            category = 'Disposable Vape'
        elif 'HOOKAH' in name_upper:
            category = 'Hookah'
        elif 'WATER PIPE' in name_upper or 'PERC' in name_upper:
            category = 'Water Pipe'
        elif 'DAB RIG' in name_upper:
            category = 'Dab Rig'
        elif 'BATTERY' in name_upper or 'MAH' in name_upper:
            category = 'Battery'
        elif 'CIGARILLO' in name_upper:
            category = 'Cigar'
        
        # Extract model (simple approach)
        if brand:
            # Remove brand from name to find model
            remaining = product_name.upper().replace(brand.upper(), '').strip()
            words = remaining.split()
            model_words = []
            for word in words[:3]:  # Take up to 3 words for model
                clean_word = word.replace('"', '').replace('″', '')
                if clean_word not in ['DISPOSABLE', 'HOOKAH', 'WATER', 'PIPE', 'DAB', 'RIG', 'PUFFS', 'PUFF']:
                    model_words.append(clean_word)
            model = ' '.join(model_words).strip()
        
        results.append({
            'Products': product_name,
            'brand': brand,
            'model': model,
            'flavor': flavor,
            'category': category,
            'puff_count': puff_count,
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
        
        # Process extraction (test with small sample)
        test_mode = True
        if test_mode:
            print("\nRunning in TEST MODE (first 100 products only)")
            df = df.head(100)
        
        print("Starting attribute extraction...")
        result_df = extract_attributes(df)
        
        # Save results
        output_file = f"extracted_attributes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        result_df.to_csv(output_file, index=False)
        print(f"\nResults saved to {output_file}")
        
        # Display results with focus on brand detection
        print("\nDetailed extraction results:")
        for i, row in result_df.iterrows():
            print(f"\nProduct: {row['Products']}")
            print(f"  Brand: '{row['brand']}' | Model: '{row['model']}' | Category: '{row['category']}'")
        
        # Summary statistics
        print(f"\n=== EXTRACTION SUMMARY ===")
        print(f"Total products processed: {len(result_df)}")
        print(f"Products with brands identified: {len(result_df[result_df['brand'] != ''])} ({len(result_df[result_df['brand'] != ''])/len(result_df)*100:.1f}%)")
        print(f"Products with models identified: {len(result_df[result_df['model'] != ''])}")
        print(f"Products with categories identified: {len(result_df[result_df['category'] != 'Other'])}")
        print(f"Products with puff counts: {len(result_df[result_df['puff_count'] != ''])}")
        
        # Brand breakdown
        print(f"\nBrand breakdown:")
        brand_counts = result_df[result_df['brand'] != '']['brand'].value_counts()
        for brand, count in brand_counts.items():
            print(f"  {brand}: {count}")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        import traceback
        traceback.print_exc()