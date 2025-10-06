import streamlit as st

st.set_page_config(
    page_title="PLUTONITA | NASA Space Apps",
    page_icon="logo/logo.jpeg",  # Puedes usar 'logo.png' si tienes el archivo local
    layout="centered"
)

st.markdown("""
    <style>
        .main { background-color: #181a1b !important; }
        .logo { width: 200px; margin-bottom: 12px; }
        .libs-row img { height: 48px; margin: 0 18px 8px 0;}
        .title { font-size: 2.8em; font-weight: bold; letter-spacing: 2px;}
        .subtitle { font-size: 1.3em; color: #eee; margin-bottom: 35px;}
        .ml-highlight { color: #f39c12; font-weight: bold;}
        .btn-start { background: #f39c12; color: white; padding: 12px 30px; border-radius: 8px; font-size: 1.1em; text-decoration: none; }
        .libs-row { margin: 22px 0 18px 0; display: flex; flex-wrap: wrap; justify-content: center;}
        .divider { height:2px; background: #444; margin:28px 0;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title" style="text-align:center;">PLUTONITA</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="text-align:center;">'
            'Unlock the Universe with <span class="ml-highlight">Machine Learning</span> 🚀<br>'
            'NASA Space Apps Project · Exoplanet Exploration'
            '</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown("""
Welcome to **Plutonita** – a modern web platform powered by **Machine Learning** and the latest data science technologies for exoplanet discovery and space exploration!

- Explore interactive datasets
- Visualize light curves and astrophysical data
- Experiment with AI-powered model

""")
st.markdown('<div style="text-align:center;"><a href="app.py" class="btn-start">Start Exploring</a></div>', unsafe_allow_html=True)#poner a donde va

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

st.markdown('<div class="libs-row" style="text-align:center;">'
    '<a href="https://streamlit.io/" title="Streamlit"><img src="https://streamlit.io/images/brand/streamlit-mark-color.png"/></a>'
    '<a href="https://numpy.org/" title="NumPy"><img src="https://upload.wikimedia.org/wikipedia/commons/3/31/NumPy_logo_2020.svg"/></a>'
    '<a href="https://pandas.pydata.org/" title="Pandas"><img src="https://pandas.pydata.org/static/img/pandas_mark.svg"/></a>'
    '<a href="https://scikit-learn.org/" title="Scikit-learn"><img src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png"/></a>'
    '<a href="https://matplotlib.org/" title="Matplotlib"><img src="https://matplotlib.org/_static/images/logo2.svg"/></a>'
    '<a href="https://lightkurve.org/" title="Lightkurve">Lightkurve - </a>'
    '<a href="https://docs.astropy.org/" title="Astropy">Astropy - </a>'
    '<a href="https://lightgbm.readthedocs.io/" title="LightGBM">LightGBM</a>'
    '</div>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center; color: #aaa; font-size:0.95em;">'
            'Powered by Python · Open Source · <a href="https://github.com/leonelcnr/SPACE-APP/" style="color:#f39c12;">View on GitHub</a>'
            '</div>', unsafe_allow_html=True)