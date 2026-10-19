"""UJ front end connected to the local calibrated EfficientNet services."""
import base64
import hashlib
import html
import io
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from cloud_runtime import get_runtime
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

st.set_page_config(page_title="UJ Wildlife Recognition", page_icon="🦒", layout="wide", initial_sidebar_state="expanded")
LOGO=Path(__file__).parent/'assets/uj_logo.jpg'
MODES={'Species classification':'classification','Species + individual identification':'identification'}

def inject_css():
    uj_orange      = "#CC3300"
    uj_orange_dark = "#A02800"
    uj_orange_light= "#FF5522"
    white          = "#FFFFFF"
    off_white      = "#FFF8F6"
    charcoal       = "#1C1C1C"
    light_gray     = "#F0F0F0"
    card_bg        = "#FFFFFF"

    st.markdown(f"""
    <style>
    /* ── Base & fonts ───────────────────────────────────────────────────── */


    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        background-color: {off_white};
        color: {charcoal};
    }}

    /* ── Remove default padding ─────────────────────────────────────────── */
    .block-container {{
        padding-top: 3.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
    }}

    /* ── HEADER BANNER ──────────────────────────────────────────────────── */
    .uj-header {{
        background: linear-gradient(135deg, {uj_orange_dark} 0%, {uj_orange} 55%, {uj_orange_light} 100%);
        padding: 1.6rem 2.5rem;
        border-radius: 0 0 18px 18px;
        display: flex;
        align-items: center;
        gap: 1.6rem;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(204,51,0,0.30);
    }}
    .uj-header-text h1 {{
        color: {white} !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin: 0 !important;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }}
    .uj-header-text p {{
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.92rem !important;
        margin: 0.3rem 0 0 0 !important;
        font-weight: 400;
    }}
    .uj-logo-img {{
        height: 88px;
        width: 88px;
        object-fit: contain;
        border-radius: 10px;
        background: rgba(255,255,255,0.12);
        padding: 6px;
        flex-shrink: 0;
    }}
    .uj-badge {{
        background: rgba(255,255,255,0.18);
        color: {white};
        border: 1px solid rgba(255,255,255,0.35);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        display: inline-block;
        margin-top: 0.45rem;
    }}

    /* ── SECTION CARDS ──────────────────────────────────────────────────── */
    .uj-card {{
        background: {card_bg};
        border-radius: 14px;
        padding: 1.6rem 1.8rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
        border: 1px solid rgba(204,51,0,0.10);
        margin-bottom: 1.2rem;
    }}
    .uj-card-title {{
        font-size: 1.05rem;
        font-weight: 700;
        color: {uj_orange};
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        border-bottom: 2px solid {off_white};
        padding-bottom: 0.7rem;
    }}

    /* ── UPLOAD ZONE ────────────────────────────────────────────────────── */
    [data-testid="stFileUploader"] > div > div {{
        border: 2.5px dashed {uj_orange} !important;
        border-radius: 12px !important;
        background: {off_white} !important;
        padding: 1.5rem !important;
        transition: all 0.2s ease;
    }}
    [data-testid="stFileUploader"] > div > div:hover {{
        border-color: {uj_orange_dark} !important;
        background: rgba(204,51,0,0.04) !important;
    }}
    [data-testid="stFileUploaderDropzoneInput"] + div {{
        color: {uj_orange} !important;
        font-weight: 600;
    }}

    /* ── BUTTONS ────────────────────────────────────────────────────────── */
    .stButton > button {{
        background: linear-gradient(135deg, {uj_orange} 0%, {uj_orange_dark} 100%) !important;
        color: {white} !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.65rem 1.8rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.2px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 3px 12px rgba(204,51,0,0.30) !important;
        width: 100%;
    }}
    .stButton > button:hover {{
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 18px rgba(204,51,0,0.42) !important;
    }}
    .stButton > button:active {{
        transform: translateY(0) !important;
    }}

    /* ── RESULT CARDS ───────────────────────────────────────────────────── */
    .result-top {{
        background: linear-gradient(135deg, {uj_orange_dark} 0%, {uj_orange} 100%);
        color: {white};
        border-radius: 14px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px rgba(204,51,0,0.30);
        text-align: center;
    }}
    .result-top .emoji {{
        font-size: 3.2rem;
        display: block;
        margin-bottom: 0.4rem;
    }}
    .result-top .label {{
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.3px;
    }}
    .result-top .conf {{
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.88;
        margin-top: 0.2rem;
    }}
    .result-top .conf-bar-wrap {{
        background: rgba(255,255,255,0.25);
        border-radius: 8px;
        height: 8px;
        margin-top: 0.8rem;
        overflow: hidden;
    }}
    .result-top .conf-bar {{
        background: {white};
        border-radius: 8px;
        height: 100%;
        transition: width 0.6s ease;
    }}

    .result-row {{
        display: flex;
        align-items: center;
        gap: 0.9rem;
        padding: 0.65rem 0;
        border-bottom: 1px solid {light_gray};
    }}
    .result-row:last-child {{ border-bottom: none; }}
    .result-rank {{
        font-size: 0.78rem;
        font-weight: 700;
        color: {uj_orange};
        width: 22px;
        text-align: center;
        flex-shrink: 0;
    }}
    .result-emoji {{ font-size: 1.3rem; flex-shrink: 0; }}
    .result-label {{
        font-size: 0.9rem;
        font-weight: 600;
        color: {charcoal};
        flex: 1;
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .result-conf {{
        font-size: 0.82rem;
        font-weight: 700;
        color: {uj_orange};
        width: 46px;
        text-align: right;
        flex-shrink: 0;
    }}
    .bar-wrap {{
        flex: 1;
        background: {light_gray};
        border-radius: 6px;
        height: 7px;
        overflow: hidden;
        min-width: 60px;
    }}
    .bar-fill {{
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, {uj_orange} 0%, {uj_orange_light} 100%);
    }}

    /* ── SIDEBAR ────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {uj_orange_dark} 0%, {uj_orange} 100%) !important;
    }}
    [data-testid="stSidebar"] * {{
        color: {white} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {{
        color: rgba(255,255,255,0.85) !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
        color: {white} !important;
        border-bottom: 1px solid rgba(255,255,255,0.3);
        padding-bottom: 0.4rem;
        font-size: 1rem !important;
    }}
    [data-testid="stSidebar"] .sidebar-info-box {{
        background: rgba(255,255,255,0.15);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.84rem;
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background: rgba(255,255,255,0.18) !important;
        color: {white} !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 8px !important;
    }}
    [data-testid="stSidebar"] .stSlider [data-testid="stTickBar"] {{
        color: rgba(255,255,255,0.6) !important;
    }}

    /* ── METRICS ROW ────────────────────────────────────────────────────── */
    .metric-box {{
        background: {white};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 3px solid {uj_orange};
    }}
    .metric-box .m-value {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {uj_orange};
    }}
    .metric-box .m-label {{
        font-size: 0.78rem;
        font-weight: 500;
        color: #666;
        margin-top: 0.15rem;
    }}

    /* ── STATUS CHIPS ───────────────────────────────────────────────────── */
    .status-ready {{
        background: #E8F5E9; color: #2E7D32;
        border: 1px solid #A5D6A7;
        border-radius: 20px; padding: 3px 12px;
        font-size: 0.78rem; font-weight: 600;
        display: inline-flex; align-items: center; gap: 4px;
    }}
    .status-pending {{
        background: #FFF3E0; color: #E65100;
        border: 1px solid #FFCC80;
        border-radius: 20px; padding: 3px 12px;
        font-size: 0.78rem; font-weight: 600;
        display: inline-flex; align-items: center; gap: 4px;
    }}

    /* ── FOOTER ─────────────────────────────────────────────────────────── */
    .uj-footer {{
        text-align: center;
        padding: 1.2rem;
        color: #999;
        font-size: 0.78rem;
        margin-top: 1rem;
        border-top: 1px solid #E8E8E8;
    }}
    .uj-footer span {{ color: {uj_orange}; font-weight: 600; }}

    /* ── HIDE default Streamlit elements ────────────────────────────────── */
    #MainMenu, footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)



def render_result(result, limit):
    if result['is_known']:
        st.markdown(f"<div class='result-top'><div class='label'>{html.escape(result['category'].title())}</div><div class='conf'>{result['confidence']*100:.4f}% species score</div></div>", unsafe_allow_html=True)
    else:
        st.warning('Unknown species — no confident match')
        st.write(f"Closest known category: **{result['closest_known_category']}**. This is a suggestion, not an accepted prediction.")
    st.caption(f"Species score {result['confidence']*100:.4f}% · Saved threshold {result['species_threshold']*100:.4f}% · Processed in {result['processing_ms']:.0f} ms")
    if result['mode']=='identification':
        st.subheader('Individual identification')
        if not result['is_known']:
            st.info('Individual matching was skipped because the species was not accepted.')
        elif result['is_known_individual']:
            st.success('Individual match: '+result['individual_id'])
        else:
            st.warning('Unknown individual — no reference match passed the saved threshold')
        if result['individual_similarity'] is not None:
            if result.get('identity_score_kind')=='adjusted_match':
                st.caption(f"Match decision score: {result['identity_score']:.4f} · Required: {result['identity_threshold']:.4f}. This score is not a probability.")
            else:
                st.caption(f"Best cosine similarity: {result['individual_similarity']:.4f} · Required: {result['identity_threshold']:.4f}. Similarity is not a probability.")
        if result['matches']:
            st.markdown('**Closest reference photos**')
            st.caption('These are suggestions for comparison. Only the first candidate can be accepted, and only when its decision score passes the threshold.')
            columns=st.columns(min(3,limit))
            for i,match in enumerate(result['matches'][:limit]):
                with columns[i%len(columns)]:
                    try:
                        st.image(get_runtime().reference_image(match['sample_id']),width='stretch')
                    except (OSError,ValueError,KeyError):
                        st.caption('Reference photo unavailable')
                    st.markdown(f"**#{match['rank']} · {match['identity']}**")
                    if 'match_score' in match:
                        st.caption(f"Ranking score {match['match_score']:.4f}")
                    else:
                        st.caption(f"Cosine similarity {match['similarity']:.4f}")
    with st.expander('Species scores'):
        st.dataframe([{'Category':p['category'],'Score (%)':round(p['confidence']*100,4)} for p in result['top_species'][:limit]],hide_index=True,width='stretch')


def main():
    inject_css()
    st.markdown("<style>[data-testid='stSidebar'] [data-baseweb='select'] * {color:#222 !important;} .uj-header {margin-top:1rem;} [data-testid='stImage'] img {max-height:420px;object-fit:contain;}</style>",unsafe_allow_html=True)
    logo=base64.b64encode(LOGO.read_bytes()).decode()
    st.markdown(f"<div class='uj-header'><img class='uj-logo-img' src='data:image/jpeg;base64,{logo}'/><div class='uj-header-text'><h1>Wildlife Classification & Identification</h1><p>Recognise the animal category. Find a matching individual.</p><span class='uj-badge'>University of Johannesburg</span></div></div>",unsafe_allow_html=True)
    try:
        with st.spinner('Loading the wildlife models…'):
            health=get_runtime().models.health
    except (OSError,ValueError,RuntimeError):
        st.error('The models could not load. Please refresh the app or contact the project team.')
        st.stop()
    with st.sidebar:
        st.image(str(LOGO),width=100)
        st.subheader('Choose a task')
        selection=st.radio('Recognition mode',list(MODES))
        mode=MODES[selection]
        limit=st.slider('Results to show',1,5,3)
        st.markdown('---')
        st.markdown('**EfficientNet-B0 · PyTorch**')
        st.write(f"{len(health['categories'])} known animal categories · 224 × 224 input")
        st.success('Both models connected')
        if mode=='identification':
            st.write(f"{health['gallery_images']:,} reference photos of {health['gallery_individuals']:,} individuals")
        st.caption('Saved recognition thresholds are fixed. Showing more results does not change the decision.')
    st.caption('Classification answers “What kind of animal?”. Identification answers “Which individual in our reference gallery?”.')
    left,right=st.columns([1,1.15],gap='large')
    with left:
        st.subheader('Upload an animal photo')
        uploaded=st.file_uploader('Choose a JPG, PNG or WebP',type=['jpg','jpeg','png','webp'],max_upload_size=8)
        content=uploaded.getvalue() if uploaded else None
        signature=(mode,hashlib.sha256(content).hexdigest() if content else None,health.get('refinement_sha256'))
        if st.session_state.get('signature')!=signature:
            st.session_state.signature=signature
            st.session_state.result=None
        valid=False
        if content:
            try:
                with Image.open(io.BytesIO(content)) as image:
                    if image.width*image.height>20_000_000:
                        st.error('Use an image smaller than 20 megapixels.')
                    else:
                        image.load()
                        st.image(ImageOps.exif_transpose(image).convert('RGB'),width='stretch')
                        st.caption(f"{uploaded.name} · {image.width} × {image.height} · {len(content)/1024:.0f} KB")
                        valid=True
            except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError):
                st.error('This image could not be read. Choose another image.')
        clicked=st.button('Classify animal' if mode=='classification' else 'Classify and identify',disabled=not valid,type='primary',width='stretch')
    with right:
        st.subheader('Recognition results')
        if clicked:
            try:
                with st.spinner('Looking at the image…'):
                    st.session_state.result=get_runtime().predict(content,mode)
            except (OSError,ValueError) as error:
                st.error(str(error))
        result=st.session_state.get('result')
        if result:
            render_result(result,limit)
        else:
            st.info('Choose a photo and run recognition to see the result.')
    with st.expander('Known categories and how matching works'):
        st.write(', '.join(health['categories']))
        st.write('The classifier uses the original deployed checkpoint. The combined mode uses a separate trained checkpoint with species and individual outputs.')
        if health.get('identity_matching')=='refined':
            st.write('For individual matching, the system compares the normal and mirrored view with reference photos of the predicted species. It reduces the influence of features shared by many animals and checks whether the first individual is a clear match compared with other individuals.')
        else:
            st.write('It compares the image embedding with reference photos of the predicted species using cosine similarity.')
        st.write('Unknown can mean an unfamiliar animal or an uncertain view. An individual can only be matched if it is represented in the reference gallery. A high score can still be wrong.')
    st.caption('Photos are processed by this hosted app and are not saved by the application. Reference photos come from the project gallery.')
    st.markdown("<div class='uj-footer'><span>University of Johannesburg</span> · Faculty of Science · Big Data Analytics · 2026</div>",unsafe_allow_html=True)

if __name__=='__main__':
    main()
