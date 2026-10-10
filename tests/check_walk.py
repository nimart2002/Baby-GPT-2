from data import make_grid, random_walk, DIRECTIONS, MOVES, GRID_SIZE

TEST_SIZE = 1000

rng = random.Random(0)
grid = make_grid(rng)

walks = {}
for i in range (TEST_SIZE):
    walks[i] = random_walk(grid, n_steps=6, rng=rng)

def test_valid():
    for i in range(TEST_SIZE):
        assert walks[i]

