# Generated by Django 3.0.4 on 2020-04-12 16:03

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fileupload', '0009_attributesinfomodel_attributesmodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemodel',
            name='attributes',
            field=models.ManyToManyField(blank=True, through='fileupload.AttributesInfoModel', to='fileupload.AttributesModel'),
        ),
    ]
