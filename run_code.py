while env.env_cls.questions_answered < 500:
    sim.step()
    if env.env_cls.time_since_last_answer > 7.0:
        print("UH OH")
        ipdb.set_trace()
