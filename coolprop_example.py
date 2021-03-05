import CoolProp.CoolProp as CP

print(CP.PropsSI("V", "T", 2500, "P", 101325, "HYDROGEN"), "Pa-s")