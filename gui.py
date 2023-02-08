import tkinter as tk
import joblib
input_window = tk.Tk()


polyuria=tk.IntVar()
polydipsia=tk.IntVar()
gender=tk.IntVar()
partial_paresis=tk.IntVar()
output_label_var=tk.StringVar()
sudden_weight_loss=tk.IntVar()
irritability=tk.IntVar()
delayed_healing=tk.IntVar()
alopecia=tk.IntVar()
itching=tk.IntVar()



polyuria_c =tk.Checkbutton(input_window, text = "polyuria", variable =polyuria , onvalue = 1, offvalue = 0, height=2,width = 20)
polydipsia_c =tk.Checkbutton(input_window, text = "polydipsia", variable =polydipsia , onvalue = 1, offvalue = 0, height=2,width = 20)
# age_c =tk.Checkbutton(input_window, text = "age", variable =age , onvalue = 1, offvalue = 0, height=2,width = 20)
age_label = tk.Label(input_window, text="age")
age_c =tk.Entry(input_window, text = "age" ,bd=5, width = 20)
gender_c =tk.Checkbutton(input_window, text = "gender(check if male)", variable =gender , onvalue = 1, offvalue = 0, height=2,width = 20)
partial_paresis_c =tk.Checkbutton(input_window, text = "partial_paresis", variable =partial_paresis , onvalue = 1, offvalue = 0, height=2,width = 20)
sudden_weight_loss_c =tk.Checkbutton(input_window, text = "sudden_weight_loss", variable =sudden_weight_loss , onvalue = 1, offvalue = 0, height=2,width = 20)
irritability_c =tk.Checkbutton(input_window, text = "irritability", variable =irritability , onvalue = 1, offvalue = 0, height=2,width = 20)
delayed_healing_c =tk.Checkbutton(input_window, text = "delayed_healing", variable =delayed_healing , onvalue = 1, offvalue = 0, height=2,width = 20)
alopecia_c =tk.Checkbutton(input_window, text = "alopecia", variable =alopecia , onvalue = 1, offvalue = 0, height=2,width = 20)
itching_c =tk.Checkbutton(input_window, text = "itching", variable =itching , onvalue = 1, offvalue = 0, height=2,width = 20)
output_label = tk.Label(input_window, textvariable=output_label_var,text="Please Input the values")



def handle_submit(event):
    # print(polyuria.get())
    # print(polydipsia.get())
    # print(age_c.get())
    # print(gender.get())
    # print(partial_paresis.get())
    # print(sudden_weight_loss.get())
    # print(irritability.get())
    # print(delayed_healing.get())
    # print(alopecia.get())
    # print(itching.get())
    output_label_var.set("Loading!!!")
    output = []
    output.append(polyuria.get())
    output.append(polydipsia.get())
    output.append(int(age_c.get())/100)
    output.append(gender.get())
    output.append(partial_paresis.get())
    output.append(sudden_weight_loss.get())
    output.append(irritability.get())
    output.append(delayed_healing.get())
    output.append(alopecia.get())
    output.append(itching.get())
    print(output)
    predict = [output,]
    loaded_model = joblib.load('rf.joblib')
    print(loaded_model)
    result = loaded_model.predict([output])
    print("the resut is:")
    print(result[0])
    if result[0] == 0:
        output_label_var.set("Congratulations you have got no Diabetes!")
    else:
        output_label_var.set("You seem to have high probability of diabetes.")


button = tk.Button(text="Check diabetes!")
button.bind("<Button-1>", handle_submit)


polyuria_c.pack()
polydipsia_c.pack()
age_label.pack()
age_c.pack()
gender_c.pack()
partial_paresis_c.pack()
sudden_weight_loss_c.pack()
irritability_c.pack()
delayed_healing_c.pack()
alopecia_c.pack()
itching_c.pack()
button.pack()
output_label.pack()


input_window.mainloop()
