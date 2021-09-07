import os

import streamlit as st
from apps import home, report, other # 新增py檔，成為頁面



#網頁設定
st.set_page_config(
    page_title="🎵傷心的人別聽慢歌",
    layout="centered",
    initial_sidebar_state="auto",
)

#左側選單Bar
class MultiApp:
    
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
       
        self.apps.append({
            "title": title,
            "function": func
        })
    def run(self):
        st.sidebar.title("🏡選單")
        app = st.sidebar.radio(
            '👇👇👇',
            self.apps,
            format_func=lambda app: app['title'])
        app['function']()
        st.sidebar.title("👨‍👨‍👦成員介紹")
        st.sidebar.info(
        "組長: AT101010 賴彥廷\n\n"
        "組員: AT101020 黃鈺凱\n\n"
        "組員: AT101045 連勁天\n\n"
        "組員: AT101018 王健安\n\n"
        )
        st.sidebar.title("📩聯繫我們")
        st.sidebar.info(
        "信箱:hi@aiacademy.tw"    
        )


app = MultiApp()

# Add new py
app.add_app("Demo", home.app)
app.add_app("Introduction", report.app)

# The main app
app.run()
