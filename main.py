import nengo
from nengo import spa

D = 64
vocab = spa.Vocabulary(D, unitary=["ONE"])
number_list = ["ONE", "TWO", "THREE", "FOUR", "FIVE",
               "SIX", "SEVEN", "EIGHT", "NINE"]

number_range = 4
vocab.parse("ONE")
for i in range(number_range):
    vocab.add(number_list[i+1], vocab.parse("%s*ONE" % number_list[i]))

model = spa.SPA(vocab)

with model:
    model.question = spa.State(D)
    model.answer = spa.State(D)
    model.count_res = spa.State(D, feedback=1)
    model.count_fin = spa.State(D, feedback=1)
    model.count_tot = spa.State(D, feedback=1)
    model.comp_tot_fin = spa.Compare(D)

    # Probably can't do this...
    actions = spa.Actions(
        # If the input isn't blank, read it in and make it blank?
        on_input="",
        # If we have finished incrementing, keep incrementing
        increment="count_res = count_res * ONE, ",
        # If we're done incrementing write it to the answer
        answer="comp_tot_fin --> answer = count_res"
    )

    model.bg = spa.BasalGanglia(actions)
    model.thal = spa.Thalamus(model.bg)

    cortical_actions = spa.Actions(
        "comp_tot_fin_A = count_fin"
        "comp_tot_fin_B = count_tot"
    )

    model.cortical = spa.Cortical(cortical_actions)