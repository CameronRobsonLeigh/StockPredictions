# Generated by Django 4.0.5 on 2022-08-13 21:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('project', '0004_bp_delete_myclass_in_model'),
    ]

    operations = [
        migrations.CreateModel(
            name='Transacs',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('userName', models.CharField(max_length=40)),
                ('amount', models.DecimalField(decimal_places=2, max_digits=6)),
                ('stockValue', models.DecimalField(decimal_places=2, max_digits=6)),
                ('prediction', models.DecimalField(decimal_places=2, max_digits=6)),
                ('stockName', models.CharField(max_length=40)),
                ('date', models.DateField()),
                ('modelType', models.CharField(max_length=40)),
            ],
        ),
    ]
