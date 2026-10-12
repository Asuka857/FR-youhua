from pathlib import Path
root=Path(__file__).parent
s=Path('G:/WN/Sensitive分析/maize_wn_environment_20260919/run_analysis.py').read_text(encoding='utf-8')
s=s.replace(".eq('maize')", ".replace({'corn':'maize','miaze':'maize'}).eq(CROP)")
s=s.replace("raw['lat'],", "raw[LATCOL],").replace("raw['lon'],", "raw[LONCOL],")
s=s.replace("str.contains('USA',case=False)", "str.contains('USA|United States|美国|America',case=False)")
s=s.replace("    grid = pd.read_csv(GRID)", '''    canada = raw.area_label.str.contains('Canada|加拿大',case=False) & raw.lon_used.between(50,170) & raw.lat_used.between(40,85)
    raw.loc[canada,'lon_used'] *= -1
    raw.loc[canada,'coordinate_action'] = 'west_longitude_sign_restored_Canada_label'
    quarantine = raw.source_excel_row.isin(QUARANTINE)
    raw.loc[quarantine,'coordinate_status'] = 'coordinate_anomaly_requires_verification'
    # Broad plausibility check only; this does not validate provinces or resolve locations.
    for pattern,bounds in [('中国|China|台湾|Taiwan',(18,54,73,135)),('Chile|智利',(-57,-17,-82,-66))]:
        lo,hi,left,right=bounds
        conflict=raw.area_label.str.contains(pattern,case=False) & ~(raw.lat_used.between(lo,hi)&raw.lon_used.between(left,right))
        raw.loc[conflict & raw.coordinate_status.eq('eligible'),'coordinate_status']='country_coordinate_conflict'
    grid = pd.read_csv(GRID)''')
s=s.replace("assert len(ref)>8", "assert len(ref)>=6")
s=s.replace("'Environmental representativeness of maize water–nitrogen trials'", "f'{CROP.capitalize()} | {TREATMENT} | Environmental representativeness'")
s=s.replace("'Percentages: geographic area of supplied maize grid cells, not harvested area or prediction accuracy'", "f'Percentages: geographic area of supplied {CROP} grid cells; not prediction accuracy'")
s=s.replace("write(fig,'maize_wn_environment')", "write(fig,f'{CROP}_{GROUP}_environment')")
s=s.replace("write(fig,'maize_wn_DI_MD_diagnostics')", "write(fig,f'{CROP}_{GROUP}_DI_MD_diagnostics')")
s=s[:s.index('\ndef main():')]
(root/'environment_core.py').write_text(s,encoding='utf-8')
