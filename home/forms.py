from django import forms
import pandas as pd
from django import forms
from .models import MyModel

Type = [
    ('line', 'line'),
    ('bar', 'bar'),
    ('scatter', 'scatter')
]

Predict_type = [
    ('ARIMA', 'ARIMA'),
    ('SARIMA', 'SARIMA'),
]

# #m = []
# for i in range(25):
#     m.append((i, i))


class MyModelForm(forms.ModelForm):
    class Meta:
        model = MyModel
        fields = ['file']


class SelectColumnForm(forms.Form):
    x_axis = forms.ChoiceField(label='X-Axis')
    y_axis = forms.ChoiceField(label='Y-Axis')
    hover_data = forms.ChoiceField(label='Additional Feature')
    plot_type = forms.ChoiceField(choices=Type, widget=forms.RadioSelect, label='Select the plot Type: ')

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns')
        super(SelectColumnForm, self).__init__(*args, **kwargs)
        self.fields['x_axis'].choices = [(col, col) for col in columns]
        self.fields['y_axis'].choices = [(col, col) for col in columns]
        self.fields['hover_data'].choices = [('None', 'None')] + [(col, col) for col in columns]


class SelectPredictForm(forms.Form):
    p_data = forms.ChoiceField(label='predict on:')
    time = forms.ChoiceField(label='predicted on:')
    prediction_type = forms.ChoiceField(choices=Predict_type, label='Select the plot Type: ')
    #seasonality = forms.ChoiceField(choices=m, label='seasonality')

    def __init__(self, *args, **kwargs):
        columns = kwargs.pop('columns')
        super(SelectPredictForm, self).__init__(*args, **kwargs)
        self.fields['p_data'].choices = [(col, col) for col in columns]
        self.fields['time'].choices = [(col, col) for col in columns]

# Read the CSV file and preprocess data
# df = pd.read_csv("static/Car_sales.xls")
# manufacturers = df['Manufacturer'].unique()
#
# Compare = [
#     ('Greater than', 'Greater than'),
#     ('Smaller than', 'Smaller than')
# ]
# Brand = [(i, i) for i in manufacturers]
#
# # Create a dictionary to map manufacturers to their models
# manufacturer_model_map = {}
# for index, row in df.iterrows():
#     if row['Manufacturer'] not in manufacturer_model_map:
#         manufacturer_model_map[row['Manufacturer']] = ['all']
#     manufacturer_model_map[row['Manufacturer']].append(row['Model'])
#
#
# class UserInputForm(forms.Form):
#     Manufacturer = forms.ChoiceField(choices=Brand, label='Manufacturer')
#     Model = forms.ChoiceField(choices=[("all", "all")], label='Model')
#     Fuel_efficiency = forms.FloatField(label='Mileage')
#     Fuel_compare = forms.ChoiceField(choices=Compare, widget=forms.RadioSelect, label='')
#     Price_in_thousands = forms.FloatField(label='Price(1000$)')
#     Price_compare = forms.ChoiceField(choices=Compare, widget=forms.RadioSelect, label='')
#
#     def __init__(self, *args, **kwargs):
#         manufacturer = kwargs.pop('manufacturer', None)
#         super(UserInputForm, self).__init__(*args, **kwargs)
#         if manufacturer:
#             models = manufacturer_model_map.get(manufacturer, ["all"])
#             self.fields['Model'].choices = [(model, model) for model in models]
#         else:
#             self.fields['Model'].choices = [("all", "all")]
