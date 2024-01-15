from streamlit_mods.links import links
from streamlit_styles.sidebar_styles import sidebar_styles as ss
import streamlit as st
#from streamlit_modules.info import info
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt


def sidebar(streamlit = st):
    yolo_feedback_contrain, contain_feedback = None, None 

    all_names = ['logo_git', 'logo_linkidin', 'git_page','linkinding_page', 'loyo_logo','my_picture', 'computer-vis']

    # get sideber style 
    custom_sidebar_style = ss()

    # initialize the style
    streamlit.write('<style>{}</style>'.format(custom_sidebar_style), unsafe_allow_html=True)
   
    # put the first image in the sidebar 
    cm          = links(name='computer-vis')
    # git hub link page 
    git_page    = links('git_page')
    # logo side bar
    side_bar_logo = plt.imread("./images/R.jpeg")
    # create image with associtated link 
    streamlit.sidebar.markdown(f'<a href="{git_page}" target="_blank"><img src="{cm}" width="360" height="200"></a>', unsafe_allow_html=True)
    
    # contains section : create the table of constains 
    streamlit.sidebar.write('<h3 class="sidebar-text">Contains</h3>', unsafe_allow_html=True)
    for i in range(2):
        streamlit.sidebar.write('', unsafe_allow_html=True)
    # list of contains 

    contains = (
        "Named Entity Recognition",
        "Neural Machine Translation",
        #"Text Summarization",
        "Question Answering With T5", 
        "Question Duplicates",
        'Sentiment Analysis', 
        'BERT & HuggingFace'
        )
    
    styles = {
        "container": {"padding": "0!important", "background-color": "#fafafa", "font-family": "Arial, sans-serif"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {
                            "background-color": "skyblue",
                            "border-radius" : "5px",
                            "margin": "3px",
                            "border": "3px solid orange",
                            "padding": "5px",
                            "width": "250px",
                            "color" : "black",
                            "font-family": "Arial, sans-serif",
                            "font-weight": "lighter",
                            "box-shadow": "2px 2px 2px 0 rgba(0, 0, 0.5, 5)"
                            },
    }
    
    icons = ('recycle', 'database-fill', 'pc-display-horizontal', 'stars', 'database', 'folder')
    # get feedback storage in contain_feedback
 
    #contain_feedback = streamlit.sidebar.radio('all contains', options=contains, disabled=False, index=None)
    with streamlit.sidebar:
        contain_feedback = option_menu('Main Menu', options=contains, menu_icon="microsoft", default_index=0,
                                       icons=icons, styles=styles)

    streamlit.write('<style>{}</style>'.format(custom_sidebar_style), unsafe_allow_html=True)
    # create 10 line of space 
    for i in range(2):
        streamlit.sidebar.write('<h5 class="author"> </h5>', unsafe_allow_html=True)
    
    # section about author
    streamlit.sidebar.write('<h3 class="sidebar-text">About Author</h3>', unsafe_allow_html=True)
    # my name 
    #streamlit.sidebar.write('<h5 class="author">Dr. Iréné Amiehe Essomba </h5>', unsafe_allow_html=True)
    # git and linkidin links apge 
    #with streamlit.sidebar.expander('', expanded=True):


    linkidin_page   = links('linkinding_page')
    # my picture took in my linkidin page 
    my_photo        = links('my_picture')
    #col1_, col2_      = streamlit.sidebar.columns(2)

    #with col1_:
    #streamlit.sidebar.markdown(f'<a class="photo" href="{linkidin_page}" target="_blank"><img src="{my_photo}"\
    #width="125" height="125" border="5px"></a>', unsafe_allow_html=True)
    st.sidebar.image(
        plt.imread("./images/image_profile.png"), 
        width=1, use_column_width="auto", clamp=True,
        caption="""Dr. Iréné Amiehe Essomba, Ph.D\n
            | Data scientist | Computer Vision expert | NLP expert"""
        )
    
    # github and likidin logo 
    #for i in range(1):
    #    streamlit.sidebar.write('<h5 class="author"> </h5>', unsafe_allow_html=True)

    logo_git        = links('logo_git')
    logo_linkidin   = links('logo_linkidin')
    email           = links('email')
    mail            = 'essomba.irene@yahoo.com'
    streamlit.sidebar.markdown(
        f'<div style="text-align: left;">'
        f'<a href="{linkidin_page}" target="_blank"><img src="{logo_linkidin}" width="30"></a>'
        f'<a href="{git_page}" target="_blank"><img src="{logo_git}" width="30"></a>'
        f'<a class="email" href="mailto:{mail}"><img class="logo"  src="{email}" width="30"></a>'
        f'</div>', 
        unsafe_allow_html=True
        )
    
    """
    for i in range(3):
        streamlit.sidebar.write('')

    streamlit.sidebar.write('<h3 class="sidebar-text">My other projets</h3>', unsafe_allow_html=True)
    vision1 = "https://vision-api.streamlit.app/"
    vision2 = "https://floraflow-api.streamlit.app/"
    path   = "./images/computer_vision.jpg"
    streamlit.sidebar.markdown(
        f'<div style="text-align: left;">'
        f'<li style="text-align: left;">'
        f'<a href="{vision1}" target="_blank"> Vision API ( Computer Vision Projet )</a>'
        f'</li>' 
        f'<li style="text-align: left;">'
        f'<a href="{vision2}" target="_blank"> FloraFlow ( Cultivate Better with AI )</a>'
        f'</li>'
        f'</div>',
        unsafe_allow_html=True
    )
    """
   
    
    return contain_feedback