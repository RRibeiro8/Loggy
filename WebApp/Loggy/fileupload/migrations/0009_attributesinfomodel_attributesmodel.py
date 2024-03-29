# Generated by Django 3.0.4 on 2020-04-12 15:07

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('fileupload', '0008_auto_20200412_1435'),
    ]

    operations = [
        migrations.CreateModel(
            name='AttributesModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tag', models.CharField(blank=True, max_length=255)),
            ],
            options={
                'ordering': ['tag'],
            },
        ),
        migrations.CreateModel(
            name='AttributesInfoModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='fileupload.ImageModel')),
                ('tag', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='fileupload.AttributesModel')),
            ],
        ),
    ]
