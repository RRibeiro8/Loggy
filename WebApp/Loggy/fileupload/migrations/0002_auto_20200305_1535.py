# Generated by Django 3.0.4 on 2020-03-05 15:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('fileupload', '0001_initial'),
    ]

    operations = [
        migrations.RenameField(
            model_name='imagemodel',
            old_name='image',
            new_name='file',
        ),
    ]
