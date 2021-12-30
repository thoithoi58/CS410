def animation(all_pops,num_gen=50):
    def update(i):
        plt.title(f'Gen {i}')
        all_pops[generation] = all_pops[generation+i]
        scatter.set_offsets(all_pops[generation])
        return scatter,

    fig = plt.figure(figsize=(15, 8))
    generation = 0
    plt.scatter(acc, flops, c='blue', label="Pareto Optimal")
    scatter = plt.scatter(
        all_pops[generation][:, 0], all_pops[generation][:, 1], s=50, c='red')

    anim = FuncAnimation(fig, update, interval=300, frames=num_gen)
    anim.save(f'imgnet.gif')