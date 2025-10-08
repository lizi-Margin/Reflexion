import re




str = "texrt #STRATEGY: 123123"
str1 = str.find("#STRATEGY:")
str_ = str[str1 + len("#STRATEGY:"):]

print(str_)

from corleone.agents.api_router import unit_test

unit_test()