from django.db import models


class MyModel(models.Model):
    file = models.FileField(upload_to='uploads/', default='uploads/default.csv')
