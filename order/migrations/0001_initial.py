# Generated by Django 3.2 on 2021-06-16 01:46

import django.core.validators
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('product', '0001_initial'),
        ('member', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Buy',
            fields=[
                ('bnum', models.AutoField(primary_key=True, serialize=False, unique=True, verbose_name='구매번호')),
                ('pnum', models.IntegerField(null=True, verbose_name='상품코드')),
                ('pname', models.CharField(max_length=100, null=True, verbose_name='상품명')),
                ('price', models.IntegerField(null=True, verbose_name='가격')),
                ('quan', models.IntegerField(null=True, verbose_name='개수')),
                ('member', models.CharField(max_length=100, null=True, verbose_name='이름')),
            ],
        ),
        migrations.CreateModel(
            name='Recommend',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('u_id', models.IntegerField(verbose_name='회원번호')),
                ('member', models.CharField(max_length=100, verbose_name='아이디')),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='product.product', verbose_name='추천상품번호')),
            ],
        ),
        migrations.CreateModel(
            name='Order',
            fields=[
                ('onum', models.AutoField(primary_key=True, serialize=False, unique=True, verbose_name='주문번호')),
                ('quan', models.PositiveSmallIntegerField(default=1, null=True, validators=[django.core.validators.MinValueValidator(1), django.core.validators.MaxValueValidator(100)], verbose_name='수량')),
                ('order_id', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='member.sign', verbose_name='이름')),
                ('prod_num', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='product.product', verbose_name='상품코드')),
            ],
        ),
    ]
