import streamlit as st

def login():

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:

        st.title("ANITS Campus AI")

        email = st.text_input(
            "Enter ANITS Email"
        )

        if st.button("Login"):

            if email.endswith("@anits.edu.in"):

                st.session_state.logged_in = True

                if "staff" in email.lower():
                    st.session_state.role = "Staff"
                else:
                    st.session_state.role = "Student"

                st.rerun()

            else:
                st.error(
                    "Only ANITS emails allowed"
                )

        st.stop()