import streamlit as st
import joblib
import google.generativeai as genai
import os


# Load your pre-trained fake news detection model
@st.cache_resource
def load_models():
    vectorizer = joblib.load("vectorizer.jb")
    model = joblib.load("lr_model.jb")
    return vectorizer, model

# Configure API key for Gemini
def configure_genai():
    # Try to get API key from different sources
    api_key = None
    
    # First check if it's stored in session state (from sidebar input)
    if 'GEMINI_API_KEY' in st.session_state:
        api_key = st.session_state.GEMINI_API_KEY
    
    # Then check environment variables
    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")
    
    # Finally try Streamlit secrets (safely)
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except:
            # No need to handle the error, we'll check api_key next
            pass
    
    if not api_key:
        st.warning("Gemini API key belum dikonfigurasi. Silakan masukkan API key di sidebar.")
        return None
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash-preview-04-17')

# Function to get explanation from Gemini
def get_explanation(news_text, is_real):
    gemini_model = configure_genai()
    if not gemini_model:
        return "Cannot provide explanation: Gemini API key not configured."
    
    label = "real" if is_real else "fake"
    
    prompt = f"""
    Analisis judul/teks berita ini sebagai seorang ahli pemeriksa fakta:
    
    "{news_text}"
    
    Model ML kami telah mengklasifikasikan ini sebagai berita {label}. 

    Berikan penjelasan singkat dan faktual (maksimal 3-4 kalimat) tentang mengapa ini mungkin berita {label}.
    Fokus pada pengidentifikasian penanda linguistik tertentu, inkonsistensi fakta, bahasa sensasional,
    atau indikator kredibilitas yang mendukung klasifikasi ini. Pastikan memberikan link seperti dari kompas.com dan lain-bila berita asli atau https://www.komdigi.go.id/berita/berita-hoaks
    bila berita itu hoaks."""
    
    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting explanation: {str(e)}"

# App UI
st.title("Detektor Berita Palsu")
st.write("Aplikasi ini menggunakan model Logistic Regression untuk mendeteksi berita palsu dengan penjelasan dari Gemini AI.")

# Sidebar for API configuration
with st.sidebar:
    st.header("Konfigurasi API")
    api_key_input = st.text_input("Gemini API Key (opsional)", type="password", 
                                help="Masukkan API key Anda di sini atau atur dalam secrets")
    
    if api_key_input:
        os.environ['GEMINI_API_KEY'] = api_key_input
    
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.write("""
    Aplikasi ini dirancang untuk membantu memprediksi apakah sebuah berita tergolong palsu atau tidak, 
    dengan menggabungkan model machine learning untuk klasifikasi berita dan teknologi AI Generative dari Gemini 
    yang memberikan penjelasan mendalam serta mudah dipahami oleh pengguna.
    """)

# Main content
news_input = st.text_area("Judul/Isi Berita:", "", height=150)

if st.button("Periksa Berita"):
    if news_input.strip():
            with st.spinner("Menganalisis berita..."):
                # Load models
                vectorizer, model = load_models()
            
                # Make prediction
                transform_input = vectorizer.transform([news_input])
                prediction = model.predict(transform_input)
                is_real = prediction[0] == 1
            
                # Display result
                if is_real:
                    st.success("✅ Berita ini kemungkinan ASLI!")
                else:
                    st.error("❌ Berita ini kemungkinan PALSU!")
            
            # Get and display explanation
            st.subheader("Penjelasan:")
            with st.spinner("Sedang membuat penjelasan..."):
                explanation = get_explanation(news_input, is_real)
                st.markdown(explanation)
            
            # Show confidence scores 
            try:
                confidence = model.predict_proba(transform_input)[0]
                st.subheader("Tingkat Kepercayaan Model:")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kemungkinan Palsu", f"{confidence[0]:.2%}")
                with col2:
                    st.metric("Kemungkinan Asli", f"{confidence[1]:.2%}")
            except:
                st.info("Model tidak memberikan skor kepercayaan.")
    else:
        st.warning("Masukkan beberapa kalimat berita untuk dianalisis...")

# Footer
st.markdown("---")
st.caption("Catatan: Aplikasi ini hanya memberikan perkiraan dan tidak menggantikan penilaian kritis. Selalu periksa sumber berita Anda.")