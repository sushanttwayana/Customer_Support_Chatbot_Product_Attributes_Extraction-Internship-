import streamlit as st
import pandas as pd
import os
from kor import create_extraction_chain
from kor.nodes import Object, Text
from langchain_openai import ChatOpenAI
from datetime import datetime
import re
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import warnings
import logging

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, filename="extraction.log", filemode="w")

load_dotenv()

# Streamlit page configuration
st.set_page_config(page_title="Automated Product Attribute Extraction")
st.title("Automated Product Attribute Extraction")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file containing product data", type=["csv"])

# Output directory
output_dir = "output-8"
os.makedirs(output_dir, exist_ok=True)

# Initialize LangChain's OpenAI wrapper for Kor
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.0,
    max_tokens=2000
)

# Kor schema for product attributes
vaping_product_schema = Object(
    id="product",
    description=(
        "Extract detailed information about vaping and smoking products from product names. "
        "Focus on accurately identifying the brand first, then category. "
        "For ambiguous cases, make reasonable inferences based on common industry terms."
    ),
    attributes=[
        Text(
            id="brand",
            description=(
                "The manufacturer or brand name. Look for trademarked brand names at the beginning of product names. "
                "If you cannot find the brand name, leave it empty or null. Do not invent brand names."
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
    ],
    examples=[
        (
            "Geek Bar Pulse X 25K Puffs Disposable Blue Raspberry (Box of 5)",
            {
                "brand": "Geek Bar",
                "category": "Disposable Vape"
            }
        ),
        (
            "EBDESIGN BC5000 DISPOSABLE 5% Nicotine",
            {
                "brand": "EBDESIGN",
                "category": "Disposable Vape"
            }
        ),
        (
            "Fruit Monster PREMIUM ELIQUID 100ML",
            {
                "brand": "Monster Vape Labs",
                "category": "E-Liquid"
            }
        ),
        (
            "7\" EVRST ART COLOR DAB RIG WATER PIPE WITH 14MM MALE BANGER",
            {
                "brand": "EVRST",
                "category": "Dab Rig"
            }
        ),
        (
            "20000 PUFFS VABEEN CYBERFLEX DISPOSABLE",
            {
                "brand": "VABEEN",
                "category": "Disposable Vape"
            }
        ),
    ],
    many=True,  # Allow multiple products in a single prompt
)

known_brands = {
    'NANO SMOKE': ['NANO SMOKE', 'NANOSMOKE'],
    'VABEEN': ['VABEEN'],
    'MYA': ['MYA'],
    'Geek Bar': ['GEEK BAR', 'GEEKBAR'],
    'Lost Mary': ['LOST MARY', 'LOSTMARY'],
    'Juicy Bar': ['JUICY BAR', 'JUICYBAR'],
    'Rookie Bar': ['ROOKIE BAR', 'ROOKIEBAR'],
    'SPACEMAN': ['SPACEMAN', 'SPACE MAN'],
    'XTRON': ['XTRON'],
    'KFBAR': ['KFBAR', 'KF BAR'],
    'EVRST': ['EVRST', 'EVEREST'],
    '3 CHI': ['3 CHI', '3CHI', 'CHI'],
    'Chris Brown': ['CHRIS BROWN'],
    'KadoBar': ['KADOBAR', 'KADO BAR'],
    'EBDESIGN': ['EBDESIGN', 'EB DESIGN'],
    'PACHAMAMA': ['Pachamama'],
    'Fruit Monster': ['FRUIT MONSTER'],
    'YOCAN': ['YOCAN'],
    'OXBAR': ['OXBAR'],
    'PILLOWZ': ['PILLOWZ'],
    'STIIIZY': ['STIIIZY'],
    'ALOHA SUN': ['ALOHA SUN'],
    'SKWEZED': ['SKWEZED'],
    'YAMI VAPORS': ['YAMI VAPORS'],
    'SUGOI VAPOR': ['SUGOI VAPOR'],
    'MAGIC MAZE': ['MAGIC MAZE'],
    'LUCY': ['LUCY'],
    'Orion Bar': ['ORION BAR', 'ORIONBAR'],
    'Coastal Cloud': ['COASTAL CLOUD'],
    'Vapetasia': ['VAPETASIA'],
    'Pod Juice': ['POD JUICE'],
    'Esco Bars': ['ESCO BARS', 'ESCO'],
    'Aloha Sun': ['ALOHA SUN'],
    'Dutch Masters': ['DUTCH MASTERS'],
    'RAMA': ['Rama', 'rama'],
    'Oak Cliff': ['oak cliff'],
    'OFF Stamp': ['OFFSTAMP', 'OFF STAMP', 'Off-Stamp'],
    'Candy King': ['Candy King', 'Candyking'],
    'Cutleaf': ['Cutleaf'],
    'MIT 45': ['Mit45', 'MIT 45'],
    'CCELL': ['CCELL', 'ccell'],
    'TRE HOUSE': ['TRE HOUSE', 'TREHOUSE'],
    'Lost Vape': ['Lost Vape', 'LostVape'],
    'RAW': ['Raw X Rolling'],
    'Lost Angel': ['Lost Angel', 'Lostangel'],
    'AL FAKHER': ['ALFAKHER', 'AL FAKHER'],
    'Pillow Talk': ['Pillow Talk', 'PillowTalk'],
    'NICE': ['nice'],
    'NEXA': ['Nexa'],
    'Camel': ['CAMEL', 'camel'],
    'SWISHER SWEETS': ['SWISHER', 'SWISHER SWEETS'],
    'BACKWOODS': ['backwood'],
    'Delata King': ['Delata King', 'Delataking'],
    'UNO MAS': ['UNO MAS', 'UNOMAS'],
    'GEEK VAPE': ['Geekvape', 'Geek Vape'],
    'DAZED 8': ['DAZED', 'DAZED 8'],
    'Adjust': ['ADJUST'],
    'Aleaf': ['Aleaf'],
    'Uwell': ['Uwell'],
    'Smok': ['Smok'],
    'Geekvape': ['Geekvape', 'Geek Vape'],
    'Voopoo': ['Voopoo'],
    'Lookah': ['Lookah'],
    'Juice Head': ['Juice Head'],
    'Naked Eliquid': ['Naked Eliquid', 'Naked 100'],
    'Tru Vapor': ['Tru Vapor', 'TruVapor'],
    'Ooze': ['Ooze'],
    'Crazy Ace': ['Crazy Ace'],
    'Caliburn': ['Caliburn'],
    'Tsunami': ['Tsunami'],
    'Grav': ['Grav'],
    'Joytech': ['Joytech', 'Joyetech'],
    'Twist': ['Twist'],
    'Tyson Vape': ['Tyson Vape', 'Tyson'],
    'LostVape Orion': ['LostVape Orion', 'LOSTVAPEORION', 'lost vape orion'],
    'UNO': ['UNO'],
    'PUFFCO': ['PUFFCO'],
    'Pax Vaporizer': ['Pax Vaporizer', 'Pax'],
    'Sili': ['Sili'],
    'URB': ['URB'],
    'Medusa': ['Medusa'],
    'Snoop Dogg Vape': ['Snoop Dogg Vape', 'Snoop Dogg'],
    'G Pen': ['G Pen'],
    'Vape Pen': ['Vape Pen'],
    'Disposables': ['Disposables'],
    'Ejuice': ['Ejuice'],
    'Grinders': ['Grinders'],
    'Detox': ['Detox'],
    'Core Infinity': ['Core Infinity', 'COREINFINITY'],
}

known_brands2 = {
    
    'Carsonator': ['Carsonator'],
    'SNICKER': ['Snicker']

}

# Function to extract brand using fallback
def extract_brand_fallback(product_name: str) -> str:
    if not isinstance(product_name, str):
        return ""
    name_upper = product_name.upper()
    for brand, variations in known_brands.items():
        for variation in variations:
            if variation in name_upper:
                return brand
    return ""

# Create Kor extraction chain
def create_kor_extraction_chain():
    try:
        chain = create_extraction_chain(
            node=vaping_product_schema,
            llm=llm,
            encoder_or_encoder_class="json",
        )
        return chain
    except Exception as e:
        st.error(f"Failed to create extraction chain: {str(e)}")
        raise e

# Process attributes using Kor with batch processing
def extract_attributes(df: pd.DataFrame, batch_size: int = 20) -> pd.DataFrame:
    st.write("Initializing extraction chain...")
    try:
        chain = create_kor_extraction_chain()
    except Exception as e:
        st.error(f"Failed to create extraction chain: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on failure
    
    results = []
    total_products = len(df)
    st.write(f"Starting extraction for {total_products} products...")
    
    # Process in batches
    for start_idx in tqdm(range(0, total_products, batch_size), desc="Processing batches"):
        end_idx = min(start_idx + batch_size, total_products)
        batch_df = df.iloc[start_idx:end_idx]
        batch_products = batch_df['Products'].tolist()
        
        try:
            # Join product names with numbering for clarity in prompt
            batch_input = "\n".join([f"{i+1}. {prod}" for i, prod in enumerate(batch_products)])
            response = chain.invoke({"text": batch_input})
            
            # Parse response
            if isinstance(response, dict):
                product_data = response.get('data', {}).get('product', [])
                if not isinstance(product_data, list):
                    product_data = [product_data]
            else:
                product_data = []
            
            # Ensure we have results for each product
            for idx, product_name in enumerate(batch_products):
                if idx < len(product_data) and isinstance(product_data[idx], dict):
                    product_info = product_data[idx]
                else:
                    product_info = {}
                
                # Extract fields with fallbacks
                brand = product_info.get('brand', '') or ''
                category = product_info.get('category', 'Other') or 'Other'
                
                # Ensure strings before stripping
                brand = str(brand).strip() if brand else ''
                category = str(category).strip() if category else 'Other'
                
                # Brand fallback
                if not brand:
                    brand = extract_brand_fallback(product_name)
                
                # Category fallback
                if category == 'Other' or not category:
                    name_upper = str(product_name).upper()
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
                
                results.append({
                    'Products': product_name,
                    'brand': brand,
                    'category': category,
                })
        except Exception as e:
            st.warning(f"Error processing batch starting at index {start_idx}: {str(e)}")
            logging.error(f"Batch error at index {start_idx}: {str(e)}\nProducts: {batch_products}")
            # Fallback for entire batch
            for product_name in batch_products:
                results.append({
                    'Products': product_name,
                    'brand': extract_brand_fallback(product_name),
                    'category': 'Other',
                })
    
    result_df = pd.DataFrame(results)
    # Ensure unique Products
    result_df = result_df.drop_duplicates(subset='Products', keep='first')
    return result_df

# Form for processing
with st.form('Process_form'):
    submitted = st.form_submit_button("Submit")

    if submitted:
        if uploaded_file is not None:
            try:
                try:
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                except UnicodeDecodeError:
                    st.warning("UTF-8 decoding failed. Trying 'latin1' encoding...")
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding='latin1')

                st.write(f"Original data: {len(df)} rows")

                # Step 1: Remove Botany products
                df = df[df['Product Category'] != 'Botany products']

                # Step 2: Remove leading whitespace from Products
                df['Products'] = df['Products'].apply(lambda x: x.lstrip() if isinstance(x, str) else '')

                # Step 3: Remove special characters
                patterns = r'[^a-zA-Z0-9-""+.()%$ ]'
                df['Products'] = df['Products'].apply(lambda x: re.sub(patterns, '', x) if isinstance(x, str) else '')

                # Step 4: Remove duplicates based on Products
                df1 = df.drop_duplicates(subset='Products', keep='first')
                df_duplicated_removed = df[df.duplicated(subset='Products')]
                st.write(f"After removing duplicates based on 'Products': {df1.shape}")
                df_duplicated_removed.to_csv(f"{output_dir}/Duplicated_based_on_products.csv", index=False)

                # Step 5: Remove products with unusual keywords
                unusual_keywords = [
                    'we stock', 'hello', 'contact', 'ADD', 'VIEW', 'LEARN', 'DISCOVER', 
                    'Elevate Your Smoke', 'Elevate Your Smoke  Vape Game',
                    'EXPLORE', 'NEW PRODUCTS', 'MARKETING', 'PARTNER', 'BECOME', 
                    'by entering this site', 'shop all brands', 'contact us', 
                    'shop now', 'marketing catalog', 'shop by', 'exotic snacks', 
                    'For Sale', 'OFFER', 'NEW ITEMS'
                ]
                filtered_df = df1[~df1['Products'].apply(lambda x: any(word.lower() in x.lower() for word in unusual_keywords) if isinstance(x, str) else False)]
                removed_unusual_df = df1[df1['Products'].apply(lambda x: any(word.lower() in x.lower() for word in unusual_keywords) if isinstance(x, str) else False)]
                st.write(f"Discarded due to unusual keywords: {len(removed_unusual_df)}")
                removed_unusual_df.to_csv(f"{output_dir}/unsual_keywords_products.csv", index=False)

                # Step 6: Fill brands using known_brands dictionary
                filtered_df = filtered_df.copy()
                def map_brand(product_name):
                    if not isinstance(product_name, str):
                        return np.nan
                    product_lower = product_name.lower()
                    for brand, aliases in known_brands.items():
                        if any(alias.lower() in product_lower for alias in aliases):
                            return brand
                    return np.nan

                filtered_df['Brand'] = filtered_df['Products'].apply(map_brand)
                no_brand_df = filtered_df[pd.isnull(filtered_df['Brand'])]
                brand_df = filtered_df[pd.notnull(filtered_df['Brand'])]

                st.write(f"Brands filled using dictionary: {len(filtered_df)}")

                # Step 7: Filter products with less than 2 words
                no_brand_df['Product_len'] = no_brand_df['Products'].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
                df_keep = no_brand_df[no_brand_df['Product_len'] > 2]
                df2_2words_products = no_brand_df[no_brand_df['Product_len'] <= 2]
                df2_2words_products.to_csv(f"{output_dir}/Products_less_then_2_words_further_processing.csv", index=False)

                df_keep.reset_index(drop=True, inplace=True)
                st.write(f"Total products for attribute extraction: {len(df_keep)}")

                # Step 8: Extract attributes using Kor
                result_df = extract_attributes(df_keep, batch_size=20)

                # Merge dictionary-filled brands with Kor-extracted results
                brand_df = brand_df.rename(columns={'Brand': 'brand'})
                brand_df['category'] = ''  # Initialize empty column for consistency
                result_df = pd.concat([result_df, brand_df[['Products', 'brand', 'category']]], ignore_index=True)

                # Remove duplicates after concatenation
                result_df = result_df.drop_duplicates(subset='Products', keep='first')

                st.write("Attributes extracted")
                st.dataframe(result_df)
                result_df.to_csv(f"{output_dir}/Extracted_attributes.csv", index=False)

                # Step 9: Map brands back to duplicated products
                brand_product_map = result_df.set_index('Products')[['brand', 'category']].to_dict('index')
                df_duplicated_removed = df_duplicated_removed.copy()
                for col in ['brand', 'category']:
                    df_duplicated_removed[col] = df_duplicated_removed['Products'].map(lambda x: brand_product_map.get(x, {}).get(col, ''))

                # Step 10: Final concatenation
                df_final = pd.concat([result_df, df_duplicated_removed[['Products', 'brand', 'category']]], ignore_index=True)

                # Save final output
                output_path = os.path.join(output_dir, f"final_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
                st.write(f"Final dataframe length: {len(df_final)}")
                df_final.to_csv(output_path, index=False)

                st.success(f"Extraction complete. File saved to {output_path}")
                st.dataframe(df_final)

            except Exception as e:
                st.error(f"Error: {e}")
                logging.error(f"General error: {str(e)}")
        else:
            st.info("Please upload a CSV file.")