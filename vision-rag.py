import streamlit as st
import cohere
import google as genai
import PIL
import numpy as np
import fitz 

st.set_page_config(page_title="Vision RAG with Cohere Embed-4", page_icon=":guardsman:", layout="wide")
st.title("Vision RAG with Cohere Embed-4 📺 ")

#Api keys
with st.sidebar:
    st.subheader("API Keys")
    cohere_api_key = st.text_input("Cohere API Key", type="password", key="")
    google_api_key = st.text_input("Google GenAI API Key", type="password", key="")

# Initialize API client
co = None
genai_client = None

if cohere_api_key and google_api_key:
    try:
        co = cohere.ClientV2(api_key=cohere_api_key)
        genai_client = genai.Client(api_key=google_api_key)
    except Exception as e:
        st.sidebar.error(f"Initialization failed, error: {e}")

##Config Ends

#Resize large images to fit model constraints
def resize_image(pil_image: PIL.Image.Image) -> None:
    org_width, org_height = pil_image.size
    max_pixels = 1568 * 1568

    if org_width * org_height > max_pixels:
        scale_factor = (max_pixels / (org_width * org_height)) ** 0.5
        new_width = int(org_width * scale_factor)
        new_height = int(org_height * scale_factor)
        pil_image.thumbnail((new_width, new_height))

#Convert images to base64 for API compatibility
def  base64_from_image(img_path: str) -> str:
    pil_image = PIL.Image.open(img_path)
    img_format = pil_image.format if pil_image.format else 'PNG'
    resize_image(pil_image)

    with io.BytesIO() as img_buffer:
        pil_image.save(img_buffer, format=img_format)
        img_buffer.seek(0)
        img_data = f"data:image/{img_format.lower()};base64, "+base64.b64encode(img_buffer.read()).decode("utf-8")
    return img_data

#Embedding generation with Cohere
@st.cache_data(ttl=3600, show_spinner=False)
def compute_image_embedding(base64_img: str, _cohere_client) -> np.ndarray | None:
    try:
        api_response = _cohere_client.embed(
            model="embed-v4.0",
            input_type="search_document",
            embedding_types=["float"],
            images=[base64_img],
        )
        if api_response.embeddings and api_response.embeddings.float:
            return np.asarray(api_response.embeddings.float[0])
        else:
            st.warning("Could not retrieve embeddings.")
            return None
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

#PDF Processing
def process_pdf_file(pdf_file, cohere_client, base_output_folder="pdf_pages") -> tuple[list[str], list[np.ndarray] | None]:
    page_image_paths = []
    page_embeddings = []
    pdf_filename = pdf_file.name
    output_folder = os.path.join(base_output_folder, os.path.splitext(pdf_filename)[0])
    os.makedirs(output_folder, exist_ok=True)

    try:
        # Open the PDF from stream
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pdf_progress = st.progress(0.0)

        for i, page in enumerate(doc.pages()):
            page_num = i+1
            page_img_path = os.path.join(output_folder, f"page_{page_num}.png")
            page_image_paths.append(page_img_path)
            #Render page to image
            pix = page.get_pixmap(dpi=150)
            pil_image= PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            pil_image.save(page_img_path, "PNG")

            #generate embedding for page image
            base64_img = pil_to_base64(pil_image)
            emb = compute_image_embedding(base64_img, _cohere_client=cohere_client)
            if emb is not None:
                page_embeddings.append(emb)
            
            #update progress
            pdf_progress.progress((i+1) / len(doc))
        
        #Filter out failed embeddings, which were marked as none
        valid_paths = [path for i, path in enumerate(page_image_paths) if i<len(page_embeddings) and page_embeddings[i] is not None]
        valid_embeddings = [emb for emb in page_embeddings if emb is not None]
        return valid_paths, valid_embeddings
    except Exception as e:
        st.error(f"Error processing PDF {pdf_filename}: {e}")
        return [], None

#Search Function
def search(question: str, co_client: cohere.Client, embeddings: np.ndarray, image_paths: list[str]) -> str | None:
    try:
        #Compute embedding for the question
        api_response = co_client.embed(
            model="embed-v4.0",
            input_type="search_query",
            embedding_types=["float"],
            texts=[question],
        )
        query_emb = np.asarray(api_response.embeddings.float[0])

        #Compute cosine similarity
        cos_sim_scores = np.dot(query_emb, embeddings.T)
        
        #Get the most relevant image
        top_idx = np.argmax(cos_sim_scores)
        hit_img_path = image_paths[top_idx]
        return hit_img_path
    except Exception as e:
        st.error(f"Error during search: {e}")
        return None

#Answer generation with Gemini
def answer(question: str, image_path: str, genai_client) -> str:
    try:
        img = PIL.Image.open(img_path)
        prompt = [f"""Answer the question based on the following image. Be as elaborate as possible giving extra relevant information.
        Don't use markdown formatting in the response. Please provide enough context for your answer.
        Question: {question}""", img]

        response = gemini_client.models.generate_content(model="gemini-2.5-flash-preview-04-17", contents=prompt)
        llm_answer = response.text
        return llm_answer
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return f"Failed to generate answer: {e}"


run_button = st.button("Run Vision RAG", key="main_run_button", 
                      disabled=not (cohere_api_key and google_api_key and question and st.session_state.image_paths and st.session_state.doc_embeddings is not None and st.session_state.doc_embeddings.size > 0))

# Output Area
st.markdown("### Results")
retrieved_image_placeholder = st.empty()
answer_placeholder = st.empty()

#Main Function RAG
#When user asks a question
if run_button:
    if co and genai_client and st.session_state.doc_embeddings is not None:
        with st.spinner("Finding relevant image..."):
            #find the most relevant image
            top_image_path = search(question, co, st.session_state.doc_embeddings, st.session_state.image_paths)
            if top_image_path:
                #Display the most relevant image
                retrieved_image_placeholder.image(top_image_path, caption=caption, use_container_width=True)
                #Generate answer
                with st.spinner("Generating answer..."):
                    final_answer = answer(question, top_image_path, genai_client)
                    answer_placeholder.markdown(final_answer)

