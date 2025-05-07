import random

def generate_pattern(length=10):
    return [random.randint(0, 1) for _ in range(length)]

def match(p1, p2, threshold):
    distance = sum([1 for a, b in zip(p1, p2) if a != b])
    return distance <= threshold

def generate_detectors(self_set, num_detectors=20, threshold=2):
    detectors = []
    while len(detectors) < num_detectors:
        candidate = generate_pattern()
        if all(not match(candidate, self_pattern, threshold) for self_pattern in self_set):
            detectors.append(candidate)
    return detectors

def classify(pattern, detectors, threshold=2):
    for detector in detectors:
        if match(pattern, detector, threshold):
            return "Damaged"
    return "Healthy"

self_patterns = [generate_pattern() for _ in range(10)]

detectors = generate_detectors(self_patterns)

test_patterns = self_patterns[:3]  
test_patterns += [generate_pattern() for _ in range(3)] 

print("=== Structure Damage Classification ===\n")
for i, pattern in enumerate(test_patterns):
    result = classify(pattern, detectors)
    print(f"Pattern {i+1}: {pattern} → {result}")

