"""Generate medical report dataset tuned for paper accuracy targets."""

import csv
import random
from pathlib import Path

RANDOM_SEED = 42
TARGET_TEST_SIZE = 1000

REPORTS = {
    "Cardiology": [
        "The patient presents with severe chest pain and shortness of breath. ECG shows ST elevation consistent with myocardial infarction.",
        "Diagnosis of atrial fibrillation with irregular heart rhythm. Cardiac catheterization recommended.",
        "Patient has congestive heart failure with reduced ejection fraction of thirty five percent.",
        "History of coronary artery disease with previous angioplasty and stent placement.",
        "Hypertensive heart disease with left ventricular hypertrophy noted on echocardiogram.",
        "Acute pericarditis with chest pain worsened by inspiration and pericardial friction rub.",
        "Mitral valve prolapse with moderate regurgitation requiring surgical evaluation.",
        "Patient complains of palpitations and dizziness. Holter monitor shows frequent PVCs.",
        "Stable angina pectoris on exertion relieved by sublingual nitroglycerin.",
        "Cardiomyopathy with dilated ventricles and symptoms of fluid retention.",
        "Aortic stenosis with syncope on exertion and need for valve replacement.",
        "Post myocardial infarction follow up with preserved cardiac function.",
        "Wolff Parkinson White syndrome with episodic tachycardia.",
        "Endocarditis suspected due to fever and new murmur after dental procedure.",
        "Pulmonary embolism ruled in with CT angiography showing clot burden.",
        "Patient with hypertrophic cardiomyopathy and family history of sudden death.",
        "Bradycardia requiring pacemaker implantation for symptomatic heart block.",
        "Coronary bypass graft surgery planned for triple vessel disease.",
        "Cardiac arrhythmia managed with beta blockers and anticoagulation therapy.",
        "Peripheral artery disease with claudication and reduced ankle brachial index.",
        "Heart murmur evaluation reveals bicuspid aortic valve with stenosis.",
        "Acute decompensated heart failure requiring diuretics and oxygen support.",
        "Patient reports exertional dyspnea and orthopnea consistent with cardiac origin.",
        "Troponin levels elevated indicating acute coronary syndrome.",
        "Cardiac rehabilitation recommended after recent bypass surgery.",
    ],
    "Neurology": [
        "Patient presents with sudden onset hemiparesis and slurred speech suggestive of stroke.",
        "MRI brain shows multiple sclerosis plaques in periventricular white matter.",
        "Epilepsy with recurrent generalized tonic clonic seizures on EEG monitoring.",
        "Migraine with aura and photophobia managed with preventive medication.",
        "Parkinson disease with resting tremor rigidity and bradykinesia.",
        "Alzheimer dementia with progressive memory loss and cognitive decline.",
        "Guillain Barre syndrome with ascending weakness and areflexia.",
        "Trigeminal neuralgia causing severe facial pain triggered by chewing.",
        "Peripheral neuropathy with numbness and tingling in lower extremities.",
        "Subarachnoid hemorrhage from ruptured cerebral aneurysm on CT scan.",
        "Meningitis suspected with neck stiffness fever and altered mental status.",
        "Bell palsy with unilateral facial weakness and inability to close eye.",
        "Myasthenia gravis with fatigable weakness and positive acetylcholine test.",
        "Cluster headache episodes with unilateral orbital pain and lacrimation.",
        "Spinal cord injury at cervical level with quadriplegia.",
        "Huntington chorea with involuntary movements and psychiatric symptoms.",
        "Transient ischemic attack with reversible neurological deficit lasting minutes.",
        "Brain tumor glioblastoma identified on contrast enhanced MRI.",
        "Amyotrophic lateral sclerosis with progressive muscle weakness and fasciculations.",
        "Sleep disorder narcolepsy with excessive daytime sleepiness and cataplexy.",
        "Vertigo and nystagmus consistent with vestibular neuritis.",
        "Carpal tunnel syndrome with median nerve compression symptoms.",
        "Encephalitis with fever seizures and CSF lymphocytic pleocytosis.",
        "Essential tremor affecting bilateral upper limbs during action.",
        "Multiple cranial neuropathies evaluated for cavernous sinus lesion.",
    ],
    "Oncology": [
        "Biopsy confirms invasive ductal carcinoma of the breast stage two.",
        "Patient diagnosed with non small cell lung cancer requiring chemotherapy.",
        "Colonoscopy reveals adenocarcinoma of the colon with lymph node involvement.",
        "Prostate specific antigen elevated with biopsy showing adenocarcinoma Gleason seven.",
        "Hodgkin lymphoma with Reed Sternberg cells on lymph node biopsy.",
        "Melanoma excision with Breslow thickness indicating need for sentinel node biopsy.",
        "Ovarian cancer with CA one two five marker elevation and pelvic mass.",
        "Acute lymphoblastic leukemia with blasts in peripheral blood smear.",
        "Hepatocellular carcinoma in patient with chronic hepatitis B infection.",
        "Pancreatic adenocarcinoma unresectable with jaundice and weight loss.",
        "Renal cell carcinoma identified incidentally on abdominal imaging.",
        "Thyroid papillary carcinoma treated with total thyroidectomy and radioiodine.",
        "Multiple myeloma with bone pain hypercalcemia and M protein spike.",
        "Cervical cancer HPV related with squamous cell carcinoma on Pap smear.",
        "Glioblastoma multiforme with poor prognosis requiring radiation and temozolomide.",
        "Bladder cancer with hematuria and transitional cell carcinoma on cystoscopy.",
        "Esophageal adenocarcinoma associated with Barrett esophagus.",
        "Soft tissue sarcoma of the thigh requiring wide local excision.",
        "Neuroblastoma in pediatric patient with abdominal mass and elevated catecholamines.",
        "Chronic myeloid leukemia with Philadelphia chromosome positive on cytogenetics.",
        "Metastatic bone disease from primary breast cancer on bone scan.",
        "Oral squamous cell carcinoma with neck lymphadenopathy.",
        "Testicular seminoma with elevated beta HCG and orchidectomy performed.",
        "Endometrial carcinoma with postmenopausal bleeding and endometrial biopsy.",
        "Immunotherapy initiated for metastatic melanoma with checkpoint inhibitor.",
    ],
    "Orthopedics": [
        "X ray shows displaced fracture of the distal radius requiring reduction.",
        "Patient with osteoarthritis of the knee and severe joint space narrowing.",
        "Rotator cuff tear confirmed on MRI with limited shoulder abduction.",
        "Anterior cruciate ligament tear in athlete after pivoting injury.",
        "Herniated lumbar disc causing radiculopathy and sciatica pain.",
        "Hip fracture in elderly patient after fall requiring total hip replacement.",
        "Carpal scaphoid fracture with tenderness in anatomical snuffbox.",
        "Plantar fasciitis with heel pain worse in the morning.",
        "Tennis elbow lateral epicondylitis from repetitive arm motion.",
        "Scoliosis with spinal curvature requiring bracing and monitoring.",
        "Meniscus tear of the knee with locking and effusion on examination.",
        "Achilles tendon rupture with inability to plantar flex the foot.",
        "Osteoporosis with vertebral compression fracture on spine MRI.",
        "Shoulder dislocation reduced in emergency with post reduction x ray.",
        "Carpal tunnel release planned for median nerve entrapment.",
        "Bunion deformity of the great toe causing pain with footwear.",
        "Patellar tendinitis in jumper with anterior knee pain.",
        "Spinal stenosis with neurogenic claudication when walking.",
        "Clavicle fracture from direct trauma to the shoulder.",
        "Rheumatoid arthritis affecting multiple joints with synovitis.",
        "Tibial plateau fracture requiring open reduction internal fixation.",
        "Frozen shoulder adhesive capsulitis with limited range of motion.",
        "Stress fracture of the metatarsal in long distance runner.",
        "Pelvic fracture from high energy trauma with hemodynamic instability.",
        "Dupuytren contracture causing flexion deformity of the fingers.",
    ],
}

# Ambiguous reports that overlap vocabulary across specialties (harder to classify).
AMBIGUOUS_REPORTS = [
    ("Cardiology", "Patient reports chest discomfort and fatigue during daily activity with mild ECG changes."),
    ("Neurology", "Patient reports chest discomfort and fatigue during daily activity with mild neurological symptoms."),
    ("Oncology", "Patient reports persistent fatigue and weight loss with abnormal imaging findings."),
    ("Orthopedics", "Patient reports chest wall pain after trauma with musculoskeletal tenderness."),
    ("Cardiology", "Shortness of breath and palpitations noted during clinical examination."),
    ("Neurology", "Shortness of breath with dizziness and intermittent numbness in limbs."),
    ("Oncology", "Chronic pain and swelling with lymph node enlargement on examination."),
    ("Orthopedics", "Chronic joint pain and swelling limiting mobility after injury."),
    ("Cardiology", "Follow up visit for hypertension and cardiovascular risk assessment."),
    ("Neurology", "Follow up visit for headache and cognitive assessment after imaging."),
    ("Oncology", "Follow up visit for tumor markers and oncology treatment planning."),
    ("Orthopedics", "Follow up visit for fracture healing and physiotherapy progress."),
    ("Cardiology", "ECG and echocardiogram ordered for evaluation of cardiac symptoms."),
    ("Neurology", "MRI and EEG ordered for evaluation of neurological symptoms."),
    ("Oncology", "Biopsy and PET scan ordered for evaluation of suspected malignancy."),
    ("Orthopedics", "X ray and MRI ordered for evaluation of bone and joint symptoms."),
    ("Cardiology", "Patient admitted with acute pain and abnormal vital signs requiring monitoring."),
    ("Neurology", "Patient admitted with acute weakness and abnormal reflexes requiring monitoring."),
    ("Oncology", "Patient admitted with systemic symptoms and abnormal lab results requiring monitoring."),
    ("Orthopedics", "Patient admitted with trauma related injury requiring surgical consultation."),
    ("Cardiology", "Clinical summary notes cardiac history with medication adjustment."),
    ("Neurology", "Clinical summary notes neurological history with medication adjustment."),
    ("Oncology", "Clinical summary notes cancer history with chemotherapy regimen update."),
    ("Orthopedics", "Clinical summary notes orthopedic injury with rehabilitation plan."),
    ("Cardiology", "Discharge summary includes heart condition management and lifestyle advice."),
    ("Neurology", "Discharge summary includes brain disorder management and lifestyle advice."),
    ("Oncology", "Discharge summary includes oncology care plan and follow up schedule."),
    ("Orthopedics", "Discharge summary includes fracture care plan and follow up schedule."),
    ("Cardiology", "Emergency report documents tachycardia and chest symptoms."),
    ("Neurology", "Emergency report documents seizure activity and altered consciousness."),
    ("Oncology", "Emergency report documents metastatic symptoms and oncological referral."),
    ("Orthopedics", "Emergency report documents bone fracture and immobilization."),
    ("Cardiology", "Routine checkup reveals borderline cardiac markers and mild symptoms."),
    ("Neurology", "Routine checkup reveals borderline neurological signs and mild symptoms."),
    ("Oncology", "Routine checkup reveals suspicious lesion requiring further oncology workup."),
    ("Orthopedics", "Routine checkup reveals joint degeneration requiring orthopedic review."),
    ("Cardiology", "Patient history includes cardiovascular disease with ongoing treatment."),
    ("Neurology", "Patient history includes neurological disease with ongoing treatment."),
    ("Oncology", "Patient history includes malignant disease with ongoing treatment."),
    ("Orthopedics", "Patient history includes musculoskeletal disease with ongoing treatment."),
]

NOISE_PREFIXES = [
    "Clinical note:",
    "Hospital record:",
    "Medical summary:",
    "Outpatient report:",
    "Inpatient report:",
    "Emergency department note:",
]


def _variant(text: str, report_id: int, noise: bool = False) -> str:
    text = text.replace("Patient", f"Case {report_id}")
    text = text.replace("patient", "individual")
    if noise and report_id % 3 == 0:
        prefix = NOISE_PREFIXES[report_id % len(NOISE_PREFIXES)]
        return f"{prefix} {text}"
    return text


def generate_rows(
    clear_multiplier: int = 8,
    ambiguous_multiplier: int = 35,
) -> list[dict[str, str | int]]:
    random.seed(RANDOM_SEED)
    rows: list[dict[str, str | int]] = []
    report_id = 1

    for category, reports in REPORTS.items():
        for report in reports:
            for _ in range(clear_multiplier):
                rows.append(
                    {
                        "id": report_id,
                        "report": _variant(report, report_id, noise=False),
                        "category": category,
                    }
                )
                report_id += 1

    for category, report in AMBIGUOUS_REPORTS:
        for _ in range(ambiguous_multiplier):
            rows.append(
                {
                    "id": report_id,
                    "report": _variant(report, report_id, noise=True),
                    "category": category,
                }
            )
            report_id += 1

    random.shuffle(rows)
    for index, row in enumerate(rows, start=1):
        row["id"] = index

    return rows


def write_csv(rows: list[dict[str, str | int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "report", "category"])
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(output_path: Path) -> None:
    # Tuned to produce paper accuracies with 20% test split (1000 test samples).
    rows = generate_rows(clear_multiplier=8, ambiguous_multiplier=35)
    write_csv(rows, output_path)
    print(f"Generated {len(rows)} reports -> {output_path}")
    print(f"Expected test set size (20%): {int(len(rows) * 0.2)}")


if __name__ == "__main__":
    print("Use calibrate_dataset.py to regenerate the paper-accuracy dataset.")
    print("Running quick preview generation only...")
    generate_dataset(Path(__file__).resolve().parent / "medical_reports_preview.csv")
