# Generated by Django 4.0.5 on 2022-08-06 13:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('project', '0003_alter_myclass_in_model_company_open_value'),
    ]

    operations = [
        migrations.CreateModel(
            name='BP',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('company_date_value', models.DateField(null=True)),
                ('company_close_value', models.FloatField(null=True)),
            ],
        ),
        migrations.DeleteModel(
            name='myClass_in_model',
        ),
    ]
