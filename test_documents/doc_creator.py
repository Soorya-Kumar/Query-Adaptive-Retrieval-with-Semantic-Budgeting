# save_separate_files.py
import json
import os

# Your test collection
test_collection = [
    # Category 1: Programming Languages (docs 0-3)
    {
        "id": "doc1",
        "title": "Python Programming",
        "text": "Python is a high-level, interpreted programming language known for its simplicity and readability. Created by Guido van Rossum in 1991, it emphasizes code readability with its notable use of significant whitespace. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming."
    },
    {
        "id": "doc2", 
        "title": "JavaScript Basics",
        "text": "JavaScript is a lightweight, interpreted programming language primarily used for web development. It enables interactive web pages and is an essential technology of the World Wide Web, alongside HTML and CSS. JavaScript supports event-driven, functional, and object-oriented programming styles."
    },
    {
        "id": "doc3",
        "title": "Java Programming",
        "text": "Java is a class-based, object-oriented programming language designed to have as few implementation dependencies as possible. It's widely used for building enterprise-scale applications, Android mobile apps, and large systems development. Java applications are typically compiled to bytecode that runs on any Java Virtual Machine."
    },
    {
        "id": "doc4",
        "title": "C++ Language",
        "text": "C++ is a powerful general-purpose programming language created by Bjarne Stroustrup. It supports both low-level memory manipulation and object-oriented programming features. C++ is widely used in systems software, game development, drivers, and performance-critical applications."
    },
    
    # Category 2: Mountains & Geography (docs 4-7)
    {
        "id": "doc5",
        "title": "Mount Everest",
        "text": "Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas. The China-Nepal border runs across its summit point. Its elevation of 8,848.86 meters was most recently established in 2020 by the Chinese and Nepali authorities."
    },
    {
        "id": "doc6",
        "title": "K2 Mountain",
        "text": "K2, also known as Mount Godwin-Austen, is the second highest mountain on Earth at 8,611 meters. It's located on the China-Pakistan border and is part of the Karakoram range. K2 is known as the 'Savage Mountain' due to its extreme difficulty and high fatality rate for climbers."
    },
    {
        "id": "doc7",
        "title": "Rocky Mountains",
        "text": "The Rocky Mountains stretch more than 4,800 kilometers from the northernmost part of British Columbia in Canada to New Mexico in the Southwestern United States. The range's highest peak is Mount Elbert in Colorado at 4,401 meters. The Rockies formed 80 to 55 million years ago during the Laramide orogeny."
    },
    {
        "id": "doc8",
        "title": "Andes Mountains",
        "text": "The Andes is the world's longest continental mountain range, stretching 7,000 kilometers along South America's western coast. The range passes through seven countries: Venezuela, Colombia, Ecuador, Peru, Bolivia, Chile, and Argentina. Its highest peak is Aconcagua at 6,961 meters."
    },
    
    # Category 3: Space & Astronomy (docs 8-11)
    {
        "id": "doc9",
        "title": "Mars Exploration",
        "text": "Mars is the fourth planet from the Sun and has been a focus of space exploration. NASA's Perseverance rover landed in 2021 to search for signs of ancient life. Mars has the largest volcano in the solar system, Olympus Mons, and evidence suggests it once had liquid water on its surface."
    },
    {
        "id": "doc10",
        "title": "Black Holes",
        "text": "A black hole is a region of spacetime where gravity is so strong that nothing can escape, not even light. They form when massive stars collapse at the end of their life cycle. The first image of a black hole was captured by the Event Horizon Telescope in 2019, showing the supermassive black hole in galaxy M87."
    },
    {
        "id": "doc11",
        "title": "Solar System",
        "text": "Our solar system consists of the Sun, eight planets, five dwarf planets, and countless asteroids and comets. The four inner planets are Mercury, Venus, Earth, and Mars, while the outer planets are Jupiter, Saturn, Uranus, and Neptune. The solar system formed 4.6 billion years ago from a giant molecular cloud."
    },
    {
        "id": "doc12",
        "title": "International Space Station",
        "text": "The International Space Station is a modular space station in low Earth orbit. It's a multinational collaborative project involving NASA, Roscosmos, JAXA, ESA, and CSA. The ISS serves as a microgravity laboratory where crews conduct experiments in biology, physics, astronomy, and meteorology."
    },
    
    # Category 4: World History (docs 12-15)
    {
        "id": "doc13",
        "title": "Roman Empire",
        "text": "The Roman Empire was founded in 27 BCE when Augustus became the first emperor. At its height, it controlled territory spanning from Britain to North Africa and from Spain to the Middle East. The empire fell in 476 CE in the west but continued as the Byzantine Empire in the east until 1453."
    },
    {
        "id": "doc14",
        "title": "World War II",
        "text": "World War II was a global conflict from 1939 to 1945 involving most of the world's nations. It was the deadliest conflict in human history with 70-85 million fatalities. The war ended with the atomic bombings of Hiroshima and Nagasaki and the establishment of the United Nations."
    },
    {
        "id": "doc15",
        "title": "Ancient Egypt",
        "text": "Ancient Egyptian civilization developed along the Nile River and lasted for over 3,000 years. Famous for pyramids, hieroglyphs, and pharaohs like Tutankhamun and Ramses II. The Egyptians made advances in mathematics, medicine, and engineering, and their culture continues to fascinate people today."
    },
    {
        "id": "doc16",
        "title": "Industrial Revolution",
        "text": "The Industrial Revolution began in Britain in the late 18th century and spread worldwide. It marked a major turning point in history with mechanization of manufacturing, new chemical processes, and the rise of the factory system. This period saw unprecedented economic growth and social change."
    },
    
    # Category 5: Technology & Computing (docs 16-19)
    {
        "id": "doc17",
        "title": "Cloud Computing",
        "text": "Cloud computing delivers computing services including servers, storage, databases, networking, and software over the Internet. Major providers include AWS, Microsoft Azure, and Google Cloud. It offers benefits like scalability, cost efficiency, and remote access from anywhere."
    },
    {
        "id": "doc18",
        "title": "Artificial Intelligence",
        "text": "Artificial intelligence involves creating machines that can think and learn like humans. Modern AI includes machine learning, deep learning, and neural networks. Applications range from voice assistants like Siri to recommendation systems on Netflix and autonomous vehicles."
    },
    {
        "id": "doc19",
        "title": "Blockchain Technology",
        "text": "Blockchain is a distributed ledger technology that maintains a secure and decentralized record of transactions. It gained prominence as the technology behind Bitcoin and other cryptocurrencies. Beyond finance, blockchain has potential applications in supply chain, voting systems, and digital identity."
    },
    {
        "id": "doc20",
        "title": "Quantum Computing",
        "text": "Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform computations. Unlike classical computers using bits (0 or 1), quantum computers use qubits that can represent multiple states simultaneously. This could revolutionize fields like cryptography and drug discovery."
    }
]

# Create a directory for the files
os.makedirs("test_documents", exist_ok=True)

# Save each document as a separate file
for doc in test_collection:
    # Create filename from id and title (sanitize title for filename)
    safe_title = "".join(c for c in doc["title"] if c.isalnum() or c in (' ', '-', '_')).replace(' ', '_')
    filename = f"test_documents/{safe_title}.txt"
    
    # Write content to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"TITLE: {doc['title']}\n")
        f.write(f"ID: {doc['id']}\n")
        f.write("-" * 50 + "\n")
        f.write(doc['text'])
    
    print(f"✅ Created: {filename}")

# Also save a metadata file for reference
metadata = [
    {
        "id": doc["id"],
        "title": doc["title"],
        "filename": f"{safe_title}.txt",
    }
    for doc in test_collection
]

with open("test_documents/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\n📁 Created {len(test_collection)} files in 'test_documents/' folder")
print("📄 Metadata saved to test_documents/metadata.json")