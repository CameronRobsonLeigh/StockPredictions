from django.shortcuts import render
from django.http import HttpResponse 
from .models import Companies
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import plotly as plotly
from io import BytesIO
import base64
import numpy as np
import time
from pmdarima.arima import auto_arima
from django.http import JsonResponse
import json
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
import plotly.graph_objs as go
from keras.models import Sequential
from keras.layers import LSTM, Dense
import datetime 
import yfinance as yf
from django.shortcuts import  render, redirect
from .forms import NewUserForm
from django.contrib.auth import login
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import AuthenticationForm 
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from django.template import loader
from .models import Transacs 
import decimal

# allow us to read the company.csv file
csv_filename2 = ""
company = ""
df = []
predictionTime = 0
close_data = []
look_back = 0
stockAndDate = []

# Create your views here.
def Index(request):
    companies = Companies.objects.all()

    context = {    
        'companies' : companies
    }
    return render(request, "pages/Home.html", context)

# transactions method
def Transactions(request):
    # grab all authenticated users transactions
    user = request.user.username
    obj = Transacs.objects.all().filter(userName=user)

    # get variety of date/times
    today = datetime.date.today()
    lastyear = datetime.date.today() - datetime.timedelta(days=5*365)
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)

    # empty list declarations
    stockAndDate = []
    storeTomorrowsDates = []
    calculateProfitPercentage = []

    # loop through transaction model
    for data in obj:
        stockAndDate.extend({data.stockName, data.date})
        tommorrowsValue = yf.download(data.stockName, start=data.date, end=data.date + datetime.timedelta(days=1), index=False)      
        tommorrowsValue = tommorrowsValue.reset_index(drop=True)       
        getTodaysValue = tommorrowsValue['Close'][-1:]       
        extractJustValue = getTodaysValue.iloc[0]
        twoDecimal = float("{:.2f}".format(extractJustValue))
        storeTomorrowsDates.append(twoDecimal)

    # return list back to front end
    mylist = zip(list(obj),storeTomorrowsDates)
    context = {    
        'obj' : mylist
    }
    return render(request, "pages/Transactions.html", context)

def GenerateGraph(df, predictionTime):
    # test size is prediction time
    TEST_SIZE = predictionTime
    train, test = df.iloc[:-TEST_SIZE], df.iloc[-TEST_SIZE:]
    x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], df.shape[0]))
    train.shape, x_train.shape, test.shape, x_test.shape

    # auto arima model
    model = auto_arima(train, start_p=1, start_q=1,
                      test='adf',
                      max_p=5, max_q=5,
                      m=1,             
                      d=1,          
                      seasonal=False,   
                      start_P=0, 
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    prediction, confint = model.predict(n_periods=predictionTime, return_conf_int=True)
    cf= pd.DataFrame(confint)

    # plot graph
    prediction_series = pd.Series(prediction,index=test.index)
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(df.Close)
    ax.plot(prediction_series)
    ax.fill_between(prediction_series.index, cf[0], cf[1],color='grey',alpha=.3)

    # return graph
    return ax.fill_between(prediction_series.index, cf[0], cf[1],color='grey',alpha=.3)

# get prediction not graph
def GeneratePrediction(df, predictionTime):
    TEST_SIZE = predictionTime
    train, test = df.iloc[:-TEST_SIZE], df.iloc[-TEST_SIZE:]
    x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], df.shape[0]))
    train.shape, x_train.shape, test.shape, x_test.shape


    model = auto_arima(train, start_p=1, start_q=1,
                      test='adf',
                      max_p=5, max_q=5,
                      m=1,             
                      d=1,          
                      seasonal=True,   
                      start_P=0, 
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)

    print(model.summary())                  
    prediction, confint = model.predict(n_periods=predictionTime, return_conf_int=True)
    cf= pd.DataFrame(confint)
    

    return prediction

# numpy encoder so that we can return numpy arrays effectively across the views
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# auto arima method that is triggered from ajax front end
def IndexCustom(request):
    # grab variety of dates
    today = datetime.date.today()
    lastyear = datetime.date.today() - datetime.timedelta(days=5*365)

    # get company that user has selected
    company = request.GET['company']

    # convert user selected company to financial stock so we can use api
    if company == "Apple":
        company = "AAPL"
    if company == "Microsoft":
        company = "MSFT"
    if company == "Netflix":
        company = "NFLX"
    if company == "Vimeo":
        company = "VMEO"
    if company == "Samsung":
        company = "005930.KS"
    if company == "BTC":
        company = "BTC-USD"

    # api to download stock data
    stock_list = [company]
    data = yf.download(stock_list, start=lastyear, end=today)

    # get todays value
    data2 = data.reset_index(drop=True)
    getTodaysValue = data2['Close'][-1:]
    extractJustValue = getTodaysValue.iloc[0]
 
    # create data frame
    companies = Companies.objects.all()
    predictionTime = request.GET['predictionTime']
    df = data   
    df.drop(['Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)
    df=df.dropna()
    row_iter = df.iterrows()

    # generate graph and prediction
    plt = GenerateGraph(df, int(predictionTime))
    prediction = GeneratePrediction(df, int(predictionTime))

    # store image of graph so we can pass to view
    buffer = BytesIO()
    plt.figure.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    # return data
    return HttpResponse(json.dumps({"graphic":graphic, "prediction": prediction, "currentValue": extractJustValue}, cls=NumpyEncoder))

# method to generate neural network
def GenerateNeuralNetwork(request):
    def predict_dates(num_prediction):
        last_date = df['Date'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates

    def predict(num_prediction, model): 
        prediction_list = close_data[-look_back:]
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            #print(x)
            x = x.reshape(1, look_back, 1)
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]

        return prediction_list

    # get forecasting time user selected
    predictionTime = request.GET['predictionTime']

    today = datetime.date.today()
    lastyear = datetime.date.today() - datetime.timedelta(days=5*365)
    company = request.GET['company']

    if company == "Apple":
        company = "AAPL"
    if company == "Microsoft":
        company = "MSFT"   
    if company == "Netflix":
        company = "NFLX"
    if company == "Vimeo":
        company = "VMEO"
    if company == "Samsung":
        company = "005930.KS"
    if company == "BTC":
        company = "BTC-USD"

    stock_list = [company]
    data = yf.download(stock_list, start=lastyear, end=today, index=False)

    data2 = data.reset_index(drop=True)
    getTodaysValue = data2['Close'][-1:]
    extractJustValue = getTodaysValue.iloc[0]

    data.insert(0, 'Date', data.index)

    df = data  
    df.drop(['Open','High','Low','Adj Close','Volume'],axis=1,inplace=True)

    close_data = df['Close'].values
    close_data = close_data.reshape((-1,1))

    # 80/20 split
    split_percent = 0.80
    split = int(split_percent*len(close_data))

    # close train/test 
    close_train = close_data[:split]
    close_test = close_data[split:]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    # 15 look back
    look_back = 15

    # tran/test generator
    train_generator = TimeseriesGenerator(close_train, close_train, length=look_back, batch_size=20)     
    test_generator = TimeseriesGenerator(close_test, close_test, length=look_back, batch_size=1)

    # LSTM model 
    model = Sequential()
    # layers
    model.add(
        LSTM(3,
            activation='relu',
            input_shape=(look_back,1))
    )
    model.add(Dense(units=4)) # hidden layer 1
    model.add(Dense(units=4)) # hidden layer 2
    model.add(Dense(units=1)) # output layer 

    # optimizer and loss function
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 25

    # fit the model
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    # predict model
    prediction = model.predict_generator(test_generator)

    # reshape data
    close_train = close_train.reshape((-1))
    close_test = close_test.reshape((-1))
    prediction = prediction.reshape((-1))

    # variables so we can pass to plotly graph
    trace1 = go.Scatter(
        x = date_train,
        y = close_train,
        mode = 'lines',
        name = 'Data'
    )
    trace2 = go.Scatter(
        x = date_test,
        y = prediction,
        mode = 'lines',
        name = 'Prediction'
    )
    trace3 = go.Scatter(
        x = date_test,
        y = close_test,
        mode='lines',
        name = 'Ground Truth'
    )
    layout = go.Layout(
        width = 700,
        height = 500
    )

    close_data = close_data.reshape((-1))
 
    # forecast days from user and forecasting 
    num_prediction = int(predictionTime) 
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    trace5 = go.Scatter(
        x = date_test,
        y = forecast,
        mode = 'lines',
        name = 'Forecast'
    )

    # return data and plot
    fig = go.Figure(data=[trace1, trace5], layout=layout)
    grabDiv = plotly.offline.plot(fig, include_plotlyjs=False, output_type='div',config = {'displayModeBar': False})
    return HttpResponse(json.dumps({"graphic": grabDiv, "prediction": forecast[1], "currentValue": extractJustValue}, cls=NumpyEncoder))

# register user function
@csrf_exempt
def register_request(request):
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save()
			login(request, user)
			messages.success(request, "Registration successful." )
			return redirect("../pages/Login")
		messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="pages/Register.html", context={"register_form":form})

# login user method
@csrf_exempt
def login_request(request):
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				messages.info(request, f"You are now logged in as {username}.")
				return redirect("../")
			else:
				messages.error(request,"Invalid username or password.")
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="pages/Login.html", context={"login_form":form})

# logout user method
@csrf_exempt
def logout_request(request):
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	return HttpResponse("ok")

# Place Transaction ajax method 
def PlaceTransaction(request):
    value = request.GET['value']
    company = request.GET['companyName']
    algorithm = request.GET['algorithm']
    currentPrediction = request.GET['currentPrediction']
    username = request.user.username
    date = datetime.date.today()

    if company == "Apple":
        company = "AAPL"
    if company == "Microsoft":
        company = "MSFT"   
    if company == "Netflix":
        company = "NFLX"
    if company == "Vimeo":
        company = "VMEO"
    if company == "Samsung":
        company = "005930.KS"
    if company == "BTC":
        company = "BTC-USD"

    today = datetime.date.today()
    lastyear = datetime.date.today() - datetime.timedelta(days=5*365)

    stock_list = company
    data = yf.download(stock_list, start=lastyear, end=today, index_col=False, header=False)

    data = data.reset_index(drop=True)
    getTodaysValue = data['Close'][-1:]
    extractJustValue = getTodaysValue.iloc[0]
    twoDecimal = float("{:.2f}".format(extractJustValue))
    
    # store in transactions database model
    Transacs.objects.create(userName = username, amount = value, stockValue = twoDecimal, prediction = currentPrediction, stockName = company, date = date, modelType = algorithm)

    return HttpResponse(request)
