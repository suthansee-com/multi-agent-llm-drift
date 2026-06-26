import streamlit as st
import json

st.set_page_config(layout="wide")
st.title("Quantifying Semantic Drift and Hallucination Accumulation in Sequential Multi-Agent LLM Architectures")
st.write("---")
st.header("choose the dataset")

file_upload = st.file_uploader("Upload dev-v2 dataset", type=["json"])
if file_upload is not None:
    try:
        data_set = json.load(file_upload)
        records = data_set.get("data", [])

        all_data = []
        for record in records:
            title = record.get("title","Unknown")
            for p_idx, para in enumerate(record.get("paras", [])):
                context = para.get("context", "").strip()
                if context:
                    all_data.append({
                        "source_text_id": f"{title}_{p_idx}",
                        "topic": title,
                        "context_text": context,
                        "char_count": len(context)
                    })

        total_records = len(all_data)
        st.success(f"Dataset loaded Sucessfully! Total available records : {total_records}")
        st.write("---")

    except Exception as e:
        st.error("Error")
else:
    st.info("Please upload the dataset")

