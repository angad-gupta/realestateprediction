from django.contrib import admin
from .models import Housing
from .models import Facility, Land


# Register your models here.
admin.site.register(Housing)
admin.site.register(Facility)
admin.site.register(Land)
#
