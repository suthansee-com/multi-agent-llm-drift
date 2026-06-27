import random
import streamlit as st
import json
import pandas as pd

st.set_page_config(layout="wide")
st.title("Quantifying Semantic Drift and Hallucination Accumulation in Sequential Multi-Agent LLM Architectures")
st.write("---")
st.header("Choose the dataset")

file_upload = st.file_uploader("Upload dev-v2 dataset", type=["json"])
if file_upload is not None:
    try:
        data_set = json.load(file_upload)
        records = data_set.get("data", [])

        all_data = []
        for record in records:
            title = record.get("title","Unknown")
            for para in record.get("paragraphs", []):
               
                for qa in para.get("qas",[]):
                    question = qa.get("question")
                    qid = qa.get("id", "")

                    answers = qa.get("answers",[])
                    answer_text = [a.get("text", "") for a in answers]
                
                    all_data.append({
                    "source_text_id": f"{title}_{qid}",
                    "topic": title,
                    "question": question,
                    "answers": answer_text,
                    "answer_count": len(answer_text)
                })

        total_records = len(all_data)
        st.success(f"Dataset loaded Sucessfully! Total available records : {total_records}")
        st.write("---")

        st.subheader("Sampling Configuration")
        col1, col2 = st.columns([1,2])

        with col1:
            sample_size = st.number_input(label="Target sample size (N):",
                                          min_value=10,
                                          max_value=min(200,total_records),
                                          value=30,
                                          step=5
                                          )
            random_seed = st.number_input("Random Seed (for reproducibility):", min_value=1, max_value=999, value=42)
            generate_btn = st.button("Generate Experiment", type="primary")

        if "batch_df" not in st.session_state:
            st.session_state["batch_df"] = None
        
        if generate_btn:
            random.seed(random_seed)
            selected_sample = random.sample(all_data, sample_size)

            st.session_state["batch_df"] = pd.DataFrame(selected_sample)
            st.toast("Batch generated successfully!")

        if st.session_state["batch_df"] is not None:
            df_batch = st.session_state["batch_df"]
        
            with col2:
                st.markdown(f"Current Pool Summery: ")
                st.write(f"Total Unique Topics Sampled: '{df_batch['topic'].nunique()}'")
                st.write(f"Average Question Length: {df_batch['question'].str.len().mean():.2f} characters")

                st.markdown("Sampled Preview")
                st.dataframe(df_batch[["source_text_id","topic","question"]], use_container_width=True, height=300)


            df_batch.to_csv("Selected_squad_batch.csv", index=False)
            st.info("Data Saved inside project dir")

    except Exception as e:
        st.error("Error")
else:
    st.info("Please upload the dataset")
