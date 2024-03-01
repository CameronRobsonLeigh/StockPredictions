from django.contrib import admin

# Register your models here.
from .models import Companies
from .models import Transacs

admin.site.register(Companies)
admin.site.register(Transacs)

