# Generated by Django 3.0.4 on 2020-03-05 17:23

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fileupload', '0009_auto_20200305_1641'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='imagemodel',
            options={'ordering': ['date_time']},
        ),
        migrations.CreateModel(
            name='LocationModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('latitude', models.FloatField(blank=True, null=True)),
                ('longitude', models.FloatField(blank=True, null=True)),
                ('name', models.CharField(blank=True, max_length=255)),
                ('timezone', models.CharField(blank=True, max_length=255)),
                ('local_time', models.DateTimeField(blank=True, null=True)),
                ('image', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='fileupload.ImageModel')),
            ],
            options={
                'ordering': ['local_time'],
            },
        ),
    ]
