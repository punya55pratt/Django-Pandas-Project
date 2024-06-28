from .forms import SelectColumnForm, SelectPredictForm
from django.http import JsonResponse
from pmdarima import auto_arima
from django.shortcuts import render, redirect
from django.urls import reverse
from .forms import MyModelForm
from .models import MyModel
import plotly.io as pio
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from io import BytesIO
import base64
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from io import StringIO


def show_data(request, data):
    # Render the display_table template with the data context
    context = {
        'data': data,
        'Topic': "Data Information",
        'text': ""
    }
    return render(request, 'display_table.html', context)


def handle_upload_file(file, request):
    df = pd.read_csv(file.path)
    header = df.columns.tolist()
    rows = df.values.tolist()
    data = (header, rows)

    # Store the data in the session for later use
    request.session['data'] = {'header': header, 'rows': rows}

    # Store the DataFrame in JSON format in the session
    request.session['dataframe'] = df.to_json()

    # Show data after upload
    return show_data(request, data)


def upload_file(request):
    if request.method == 'POST':
        form = MyModelForm(request.POST, request.FILES)
        if form.is_valid():
            instance = form.save()
            return handle_upload_file(instance.file, request)
    else:
        form = MyModelForm()
    return render(request, 'upload.html', {'form': form})


def select_columns(request):
    data = request.session.get('data')
    if not data:
        return redirect('upload')

    if request.method == 'POST':
        form = SelectColumnForm(request.POST, columns=data['header'])
        if form.is_valid():
            x_axis = form.cleaned_data['x_axis']
            y_axis = form.cleaned_data['y_axis']
            hover_data = form.cleaned_data['hover_data']
            plot_type = form.cleaned_data['plot_type']

            request.session['decomposed_plot'] = {'y_axis': y_axis, 'x_axis': x_axis}

            plot_url = f"/plot/{x_axis}/{y_axis}/{plot_type}/{hover_data}"
            return redirect(plot_url)
    else:
        form = SelectColumnForm(columns=data['header'])

    return render(request, 'select_columns.html', {'form': form})


def plot(request, x_axis, y_axis, plot_type, hover_data):
    data = request.session.get('data')
    if not data:
        return redirect('upload')

    df = pd.DataFrame(data['rows'], columns=data['header'])
    pio.templates.default = "plotly_dark"
    plot_functions = {
        'line': px.line,
        'bar': px.bar,
        'scatter': px.scatter
    }

    if plot_type in plot_functions and hover_data != 'None':
        fig = plot_functions[plot_type](df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}', color=hover_data)

    elif plot_type in plot_functions and hover_data == 'None':
        fig = plot_functions[plot_type](df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')
    else:
        fig = px.line(df, x=x_axis, y=y_axis, title=f'{y_axis} vs {x_axis}')

    plot_html = fig.to_html(full_html=False)

    return render(request, 'plot.html', {'plot_html': plot_html})


def Predict(request):
    data = request.session.get('data')
    if not data:
        return redirect('upload')

    if request.method == 'POST':
        form = SelectPredictForm(request.POST, columns=data['header'])
        if form.is_valid():
            p_data = form.cleaned_data['p_data']
            time = form.cleaned_data['time']
            prediction_type = form.cleaned_data['prediction_type']
            #seasonality = form.cleaned_data['seasonality']

            #request.session['prediction_data'] = {'p_data': p_data, 'time': time, 'prediction_type': prediction_type,
            #                                        'seasonality': seasonality}

            predict_url = f"/predict/{p_data}/{time}/{prediction_type}"
            return redirect(predict_url)
    else:
        form = SelectPredictForm(columns=data['header'])

    return render(request, 'predict.html', {'form': form})


def Prediction_plot(request, p_data, time, prediction_type):
    data = request.session.get('data')
    if not data:
        return redirect('upload')

    df = pd.DataFrame(data['rows'], columns=data['header'])

    if prediction_type == 'ARIMA' or prediction_type == 'SARIMA':
        flag = 0
        try:
            df['mod_time'] = pd.to_datetime(df[time])
            df = df.drop(time, axis=1)
            flag = 1
            df = df.set_index('mod_time')
        except (ValueError, TypeError) as e:
            print(f"Error converting '{time}' to datetime: {e}")
            return render(request, 'error.html', {'message': f"Error converting '{time}' to datetime: {e}"})

        df_decomposed = seasonal_decompose(df[p_data], model='multiplicative')
        seasonality = 0
        for i in range(1, len(df_decomposed.seasonal)):
            if df_decomposed.seasonal.iloc[i] == df_decomposed.seasonal.iloc[0]:
                seasonality = i
                break

        if flag == 1 and prediction_type == 'SARIMA':
            stepwise_fit = auto_arima(df[p_data], start_p=0, start_q=0, max_p=3, max_q=3, start_P=0, m=seasonality,
                                      d=None, D=1, seasonal=True, stepwise=True, trace=True)

            order = stepwise_fit.order
            seasonal_order = stepwise_fit.seasonal_order
            train = df[:len(df) - seasonality]
            test = df[len(df) - seasonality:]
            model = SARIMAX(train[p_data], order=order, seasonal_order=seasonal_order)
            result = model.fit()

            test_1 = result.predict(start=len(train), end=len(df)-1).rename("Test")

            forecast = result.predict(start=len(df),
                                      end=(len(df) - 1) + 3 * seasonality).rename("Forecast")

            actual_test_values = test[p_data]

            print(len(test_1))
            print(len(actual_test_values))
            assert len(test_1) == len(
                actual_test_values), "Prediction and actual test values lengths do not match"

            # Calculate performance metrics
            mse = mean_squared_error(actual_test_values, test_1)
            rmse = np.sqrt(mse)

            print(f'Mean Squared Error: {mse}')
            print(f'Root Mean Squared Error: {rmse}')

            print("Seasonality::::",seasonality)
            # fig, ax = plt.subplots(figsize=(14, 7))
            # df[p_data].plot(ax=ax, label='Observed', color='blue')
            # forecast.plot(ax=ax, label='Forecast', color='red')
            # ax.set_xlabel('Time')
            # ax.set_ylabel(p_data)
            # ax.set_title(f'Forecasted Value {p_data} with {time}')
            # ax.legend()
            # plt.tight_layout()
            fig = px.line(df, x=df.index, y=p_data, title=f'Forecasted Value {p_data} with {time}')
            fig.add_scatter(x=test_1.index, y=test_1, mode='lines', name='Test Prediction', line=dict(color='green'))
            fig.add_scatter(x=forecast.index, y=forecast, mode='lines', name='Forecast', line=dict(color='red'))

            # Convert plot to base64 encoded string
            # buffer = BytesIO()
            # plt.savefig(buffer, format='png')
            # buffer.seek(0)
            # plot_base64 = base64.b64encode(buffer.getvalue()).decode()
            # plt.close()
            plot_html = fig.to_html(full_html=False)

            return render(request, 'prediction_plot.html', {'plot_html': plot_html})

    # Handle other prediction types or return a default response


def decompose(request):
    decomposed_plot = request.session.get('decomposed_plot')
    data = request.session.get('data')
    if not data:
        return redirect('upload')

    if not decomposed_plot:
        return redirect('plot')

    df = pd.DataFrame(data['rows'], columns=data['header'])
    df_decomposed = seasonal_decompose(df[decomposed_plot['y_axis']], model='multiplicative')

    fig, ax = plt.subplots(figsize=(14, 7))
    df_decomposed.plot(ax=ax)
    ax.set_xlabel(decomposed_plot['x_axis'])
    ax.set_ylabel(decomposed_plot['y_axis'])
    ax.legend()
    plt.tight_layout()

    # Convert plot to base64 encoded string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()

    return render(request, 'prediction_plot.html', {'plot_base64': plot_base64})

# def update_models(request):
#     manufacturer = request.GET.get('manufacturer')
#     models = manufacturer_model_map.get(manufacturer, [("all", "all")])
#     return JsonResponse({'models': models})
#
#
# def user_input(request):
#     data = None
#     manufacturer = None
#
#     if request.method == 'POST':
#         manufacturer = request.POST.get('Manufacturer')
#         form = UserInputForm(request.POST, manufacturer=manufacturer)
#         if form.is_valid():
#             Manufacturer = form.cleaned_data['Manufacturer']
#             Model = form.cleaned_data['Model']
#             Fuel_efficiency = form.cleaned_data['Fuel_efficiency']
#             Fuel_compare = form.cleaned_data['Fuel_compare']
#             Price_in_thousands = form.cleaned_data['Price_in_thousands']
#             Price_compare = form.cleaned_data['Price_compare']
#
#             # Read the CSV file
#             csv_path = 'static/Car_sales.xls'
#             df = pd.read_csv(csv_path)
#
#             # Data preprocessing
#
#             # Apply user filters
#             preferred_df = df.copy()
#             if Manufacturer != "all":
#                 preferred_df = preferred_df[preferred_df['Manufacturer'] == Manufacturer]
#             if Model != "all":
#                 preferred_df = preferred_df[preferred_df['Model'] == Model]
#             if Fuel_efficiency is not None and Fuel_compare == 'Greater than':
#                 preferred_df = preferred_df[preferred_df['Fuel_efficiency'] >= Fuel_efficiency]
#             if Fuel_efficiency is not None and Fuel_compare == 'Smaller than':
#                 preferred_df = preferred_df[preferred_df['Fuel_efficiency'] <= Fuel_efficiency]
#             if Price_in_thousands is not None and Price_compare == 'Greater than':
#                 preferred_df = preferred_df[preferred_df['Price_in_thousands'] >= Price_in_thousands]
#             if Price_in_thousands is not None and Price_compare == 'Smaller than':
#                 preferred_df = preferred_df[preferred_df['Price_in_thousands'] <= Price_in_thousands]
#
#             # Convert data to a format suitable for the template
#             header = preferred_df.columns.tolist()
#             rows = preferred_df.values.tolist()
#             data = (header, rows)
#
#             context = {
#                 'data': data,
#                 'Topic': "Car Information",
#                 'text': "According to Your Search"
#             }
#
#             return render(request, 'display_table.html', context)
#     else:
#         form = UserInputForm()
#
#     return render(request, 'user_input.html', {'form': form})
#
#
# def index(request):
#     df = pd.read_csv('static/Car_sales.xls')
#     # Perform operations on the DataFrame
#     # Example: Filter rows where 'Column1' > 50 and create a new column 'Sum'
#     # Convert DataFrame to list of lists for template rendering
#     header = df.columns.tolist()
#     rows = df.values.tolist()
#     data = (header, rows)
#     context = {'data': data,
#                'Topic': "Car Information",
#                'text': ""}
#     return render(request, 'display_table.html', context)
#
#
# def show(request):
#     df = pd.read_csv('static/Car_sales.xls')
#     df_new = modified(df)
#     header = df_new.columns.tolist()
#     rows = df_new.values.tolist()
#     data = (header, rows)
#     context = {'data': data,
#                'Topic': "Car Information",
#                'text': ""}
#     return render(request, 'display_table.html', context)
#
#
# def modified(df):
#     df = df[df['Make'] == "Honda"]
#     return df
def basic_dataframe_analysis(request):
    if 'dataframe' in request.session:
        df_json = request.session['dataframe']
        df = pd.read_json(df_json)
        analysis_results = perform_basic_analysis(df)
        return render(request, 'analysis.html', {'analysis_results': analysis_results})
    else:
        return JsonResponse({'error': 'No DataFrame found in session'}, status=400)

def perform_basic_analysis(df):
    analysis_results = {}

    print("Dataframe:::::\n",df)
    print("Dataframe Head:::::\n", df.head())
    # Display the first few rows of the DataFrame
    analysis_results['head'] = df.head().to_dict(orient='list')

    # Summary statistics
    summary_stats = df.describe(include='all').reset_index()
    summary_stats = summary_stats.melt(id_vars=['index']).sort_values(by=['variable', 'index'])
    summary_stats.columns = ['Statistic', 'Column', 'Value']
    analysis_results['summary_statistics'] = summary_stats.to_dict(orient='records')

    # Data types of each column
    analysis_results['data_types'] = df.dtypes.astype(str).to_dict()

    # Missing values in each column
    analysis_results['missing_values'] = df.isnull().sum().to_dict()

    # Basic information about the DataFrame
    buf = StringIO()
    df.info(buf=buf)
    buf.seek(0)
    analysis_results['info'] = buf.getvalue()

    return analysis_results


