# Generated by Django 3.0.4 on 2020-04-12 12:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fileupload', '0005_imagemodel_categories'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='locationinfomodel',
            options={},
        ),
        migrations.AlterField(
            model_name='locationmodel',
            name='tag',
            field=models.CharField(blank=True, max_length=255),
        ),
    ]