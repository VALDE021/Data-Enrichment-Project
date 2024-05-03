import streamlit.components.v1 as components 
# Source: https://meta.stackoverflow.com/questions/392785/need-to-add-linkedin-and-github-badges-in-profile-page-of-stack-overflow
links_html = """<ul>
<li><a href="https://github.com/VALDE021/Movies-IMDB-Project/tree/main">Project Repo</a></li>
  <li><a href="https://www.linkedin.com/in/eric-n-valdez-94a9003/[removed]" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/gVE0j.png" alt="linkedin"> LinkedIn
  </a> </li>
  <li><a href="https://github.com/VALDE021" rel="nofollow noreferrer">
    <img src="https://i.stack.imgur.com/tskMh.png" alt="github"> Github
  </a></li>
</ul>"""
components.html(links_html)
st.markdown(contact_info)


