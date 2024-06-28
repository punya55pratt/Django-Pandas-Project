from django.urls import path
from .views import upload_file, select_columns, plot, Predict, Prediction_plot, basic_dataframe_analysis

urlpatterns = [
    path('', upload_file, name='upload'),
    path('analyze/', basic_dataframe_analysis, name='analyze'),
    path('select_columns/', select_columns, name='select_col'),
    path('predictions/', Predict, name='Predict'),
    path('predict/<str:p_data>/<str:time>/<str:prediction_type>', Prediction_plot,
         name='prediction'),
    path('plot/<str:x_axis>/<str:y_axis>/<str:plot_type>/<str:hover_data>', plot, name='plot'),
]
