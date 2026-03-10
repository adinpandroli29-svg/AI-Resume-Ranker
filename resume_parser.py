import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text


if __name__ == "__main__":

     job_description = """
    Looking for a software developer with experience in programming,
    data analysis, and database management. The candidate should have
    knowledge of Python, Java, C++, SQL, and data analysis libraries
    such as Pandas and NumPy. Experience with machine learning,
    data science, and problem-solving skills is preferred.
    """  
     
     resume_folder = "resumes"
     resume_files = [
    os.path.join(resume_folder, file)
    for file in os.listdir(resume_folder)
    if file.endswith(".pdf")
    ]
     
     skills = [
    "python","java","c++",
    "machine learning","deep learning",
    "data analysis","data science",
    "sql","mysql","postgresql",
    "pandas","numpy","matplotlib",
    "tensorflow","keras","scikit-learn",
    "power bi","tableau","excel"
    ]
     
     resumes = [extract_text(r) for r in resume_files]
     print("\nDetected Skills in Resumes:\n")

     for i, resume_text in enumerate(resumes):
         detected = []

         for skill in skills:
             if skill.lower() in resume_text.lower():
              detected.append(skill)

         print(f"{os.path.basename(resume_files[i])}: {detected}")


     documents = resumes + [job_description]
     vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
     tfidf_matrix = vectorizer.fit_transform(documents)
     job_vector = tfidf_matrix[-1]

     scores = cosine_similarity(tfidf_matrix[:-1], job_vector)

     results = []

     for i, score in enumerate(scores):
      results.append((os.path.basename(resume_files[i]), score[0]*100))

# sort resumes by score
     results.sort(key=lambda x: x[1], reverse=True)

     print("\nResume Ranking:\n")

     for rank, (file, score) in enumerate(results, 1):
      print(f"{rank}. {file} — {score:.2f}%")

     best_index = scores.argmax()
     best_score = scores[best_index][0] * 100

     print("\nBest Resume:", os.path.basename(resume_files[best_index]))
     print("Best Match Score:", f"{best_score:.2f}%")