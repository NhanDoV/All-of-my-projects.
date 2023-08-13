from flask import Flask, render_template, request
import pandas as pd
app = Flask(__name__)

table = pd.DataFrame({'Bậc lương': range(1,8), 
                        'mức thuế': ['Đến 05 hay (0, 5) triệu đồng', '[5, 10] triệu', '[10, 18] triệu', '[18, 32] triệu', '[32, 52] triệu', '[52, 80] triệu', '> 80 triệu'],
                        'thuế suất': [f"{k}%" for k in range(5, 40, 5)]
                    })

@app.route('/')
def home():
    return render_template('home.html', table=table)

@app.route('/', methods=['POST'])

def main():
    gross_income = float(request.form['salary'])
    nb_depends = float(request.form['nb_dependencies'])
    insurances = float(request.form['bhxh'])

    print(f"{gross_income}, {nb_depends}, {insurances}")
    print(f"{type(gross_income)}, {type(nb_depends)}, {type(insurances)}")

    # compute
    gross = gross_income - (11 + insurances + nb_depends*4.4)

    # Argument
    if gross < 5:
        tax = gross * 0.05
        tax_lv = 5
    elif gross < 10:
        tax = 5*0.05 + (gross - 5)*0.1
        tax_lv = 10
    elif gross < 18:
        tax = 5*0.05 + (10-5)*0.1 + (gross - 10)*0.15
        tax_lv = 15
    elif gross < 32:
        tax = 5*0.05 + (10-5)*0.1 + (18-10)*0.15 + (gross - 18)*0.2
        tax_lv = 20
    elif gross < 52:
        tax = 5*0.05 + (10-5)*0.1 + (18-10)*0.15 + (32-18)*0.2 + (gross - 32)*0.25
        tax_lv = 25
    elif gross < 80:
        tax = 5*0.05 + (10-5)*0.1 + (18-10)*0.15 + (32-18)*0.2 + (52-32)*0.25 + (gross - 52)*0.3
        tax_lv = 30
    else:
        tax = 5*0.05 + (10-5)*0.1 + (18-10)*0.15 + (32-18)*0.2 + (52-32)*0.25 + (80-52)*0.3 + (gross - 80)*0.35
        tax_lv = 35

    net_income = gross_income - tax

    return(render_template('home.html',
                            table = table,
                            tax = round(tax, 3),
                            ins = insurances,
                            gross = round(gross, 3),
                            gross_init = round(gross_income, 3),
                            sub_dep = round(nb_depends*4.4, 3),
                            net_inc = round(net_income, 3),
                            tax_level = tax_lv
                        )
            )    

if __name__ == "__main__":
    app.run(port='5050', threaded=False, debug=True)