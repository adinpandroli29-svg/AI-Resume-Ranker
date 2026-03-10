import streamlit as st
import plotly.express as px
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


st.title("AI Resume Ranker")

job_description = st.text_area("Enter Job Description")

uploaded_files = st.file_uploader(
    "Upload Resumes", type="pdf", accept_multiple_files=True
)

# skills database
skills = [
    "python","java","c++",
    "machine learning","deep learning",
    "data analysis","data science",
    "sql","mysql","postgresql",
    "pandas","numpy","matplotlib",
    "tensorflow","keras","scikit-learn",
    "power bi","tableau","excel"
]

if st.button("Analyze"):

    if not uploaded_files or job_description == "":
        st.warning("Please upload resumes and enter job description")

    else:
        resumes = []
        filenames = []

        for file in uploaded_files:
            text = extract_text(file)
            resumes.append(text)
            filenames.append(file.name)

        documents = resumes + [job_description]

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        tfidf_matrix = vectorizer.fit_transform(documents)

        job_vector = tfidf_matrix[-1]
        scores = cosine_similarity(tfidf_matrix[:-1], job_vector)

        results = []

        for i, score in enumerate(scores):
            results.append((filenames[i], resumes[i], score[0]*100))

        results.sort(key=lambda x: x[2], reverse=True)

        st.subheader("Resume Ranking  Visualization")

        for rank, (file, text, score) in enumerate(results, 1):

            detected = []
            missing = []

            for skill in skills:
                if skill in text.lower():
                    detected.append(skill)

                if skill in job_description.lower() and skill not in text.lower():
                    missing.append(skill)

            st.markdown(f"### {rank}. {file}")
            st.write(f"Score: {score:.2f}%")

            st.write("Matched Skills:", detected)

            st.write("Missing Skills:", missing)

            st.write("---")


        # Chart Visualization
        scores_chart = [score for _, _, score in results]
        names_chart = [file for file, _, _ in results]
        
        fig = px.bar(
            x=names_chart,
            y=scores_chart,
            labels={"x": "Resume", "y": "Match Score (%)"},
            title="Resume Match Score Comparison"
            )
        st.plotly_chart(fig, key="resume_chart")

        st.success(f"Best Resume: {results[0][0]} ({results[0][2]:.2f}%)")