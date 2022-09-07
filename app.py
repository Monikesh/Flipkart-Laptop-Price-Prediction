import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open('pipe3.pkl','rb'))
df = pickle.load(open('df3.pkl','rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand',df['ProductName'].unique())
#
star = st.number_input('Rating-Star')

#cpu
cpu = st.selectbox('CPU',df['Processor'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# screen size
ScreenSize = st.number_input('Screen Size(in cm)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])


hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])


os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

  
    query = np.array([company,star,cpu,ram,ScreenSize,touchscreen,hdd,ssd,os])

    query = query.reshape(1,9)
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))

