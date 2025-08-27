import streamlit as st
import pandas as pd
import os
from pydantic import BaseModel, Field
from typing import Annotated, List
from langchain_core.output_parsers import PydanticOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import ast

load_dotenv()

st.set_page_config(page_title="Automated feature extraction")
st.title("Automated feature extraction")

# Sidebar
with st.sidebar:
    st.header("Only upload a CSV File")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

output_dir = "Outputs"
os.makedirs(output_dir, exist_ok=True)

class AttributeExtractor(BaseModel):
    brand : Annotated[str , Field(..., description="Brand of the product",examples=["Geek Bar",'Fruit Monster','Candy King','Rama','Cutleaf','MIT 45','SMOK','Coastal Cloud','CCELL'])]
    model : Annotated[str , Field(..., description="Model of the product")]

# LLM + parser setup
def create_parser():
    # llm = ChatOpenAI(model_name="gpt-4o")
    llm = ChatGroq(model_name = "llama-3.3-70b-Versatile",max_tokens= 1200) # Load the Groq Model
    parser = PydanticOutputParser(pydantic_object=AttributeExtractor)
    fixing_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
    return llm, fixing_parser

# Prompt template
def create_prompt():
    template = """Extract the following fields from each vape product description:
- "brand" (examples are: "Geek Bar",'Fruit Monster','Candy King','Rama','Cutleaf','MIT 45','SMOK','Coastal Cloud','CCELL','TRE HOUSE','AL FAKHER)
- "model" (examples are : "Premium" , "Pulse" , "" ,"BC5000", "" )

If a value is not present, respond with "n/a".

Return ONLY a single valid JSON array like this:
[
  {{ "brand": "GEEK BAR", "model": "PULSE" }},
  {{ "brand": "CCELL", "model": "Go Stik" }},
  {{ "brand": "n/a", "model": "n/a" }}
]

Descriptions:
{descriptions}
"""
    return PromptTemplate(input_variables=["descriptions"],  template=template)


# Batch LLM call
def process_batch(llm, parser, descriptions):
    joined_input = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(descriptions)])
    prompt = create_prompt()
    final_prompt = prompt.format_prompt(descriptions=joined_input).to_string()
    # print(final_prompt)
    result = llm.invoke(final_prompt)

    try:
        st.write(type(result.content))
        parsed = ast.literal_eval(result.content)
        return parsed
    except Exception as e:
        st.warning(f"Failed to parse batch response:\n{result.content}")
        return [{"brand": "n/a", "model": "n/a"} for _ in descriptions]

# Form
with st.form('Process_form'):
    submitted = st.form_submit_button("Submit")

    if submitted:
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Original data: {len(df)} rows")
                df.drop_duplicates(subset=['Products'],inplace=True)
                st.write(f"Data after removing duplicate values: {len(df)} rows")

                df['product_len'] = df['Products'].apply(lambda x: len(str(x).split()))
                df_discard = df[
                    (df['product_len'] <= 2) |
                    (df['website'] == 'www.rrrwholesale.com') |
                    (df['Product Category'] == "Apparels")
                ]
                df_keep = df[
                    (df['product_len'] > 2) &
                    (df['website'] != 'www.rrrwholesale.com') &
                    (df['Product Category'] != "Apparels")
                ].copy()

                st.write(f"Discarded: {len(df_discard)} | Kept: {len(df_keep)}")
                df_keep.reset_index(drop=True, inplace=True)
                df_keep["brand"] = ""
                df_keep["model"] = ""

                llm, parser = create_parser()

                batch_size = 10
                for i in tqdm(range(0, len(df_keep), batch_size), desc="Batch Processing"):
                    batch_df = df_keep.iloc[i:i+batch_size]
                    descriptions = batch_df["Products"].tolist()
                    results = process_batch(llm, parser, descriptions)
                    print(results)

                    for j, item in enumerate(results):
                        idx = i + j
                        df_keep.at[idx, "brand"] = item.get("brand", "n/a")
                        df_keep.at[idx, "model"] = item.get("model", "n/a")

                # Save result
                output_path = os.path.join(output_dir, "final_output_batch.csv")
                df_keep.to_csv(output_path, index=False)
                st.success(f"Extraction complete. File saved to {output_path}")
                st.dataframe(df_keep)

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("Please upload a CSV file.")
