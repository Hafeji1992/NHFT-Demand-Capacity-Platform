
        -- ============================================================================
        -- PATIENT DATA EXTRACTION QUERY
        -- ============================================================================
        -- Purpose: Extract Patient-related metrics from waiting times and
        -- contact attendance views, pulling referrals, contacts, waiters, caseload,
        -- discharge, and treatment metrics.
        -- ============================================================================
        SELECT
            -- Identifiers
            WT.[ProviderCodeCurrent],
            WT.[Service_Line],
            WT.[PeriodEnd],

            -- ======================================================================== 
            -- REFERRAL AND CLOCK STOP METRICS
            -- ========================================================================
            WT.[Referrals],
            WT.[ClockStopActuals],

            -- ======================================================================== 
            -- WAITER METRICS
            -- ========================================================================
            WT.[Waiters],
            WT.[WaitersOver18Weeks],

            -- ======================================================================== 
            -- CASELOAD METRICS
            -- ========================================================================
            WT.[Caseload],
            CA.[TotalCaseloadContacts],
            CA.[FTFCaseloadContacts],

            -- ======================================================================== 
            -- CONTACT METRICS
            -- ========================================================================
            CA.[TotalContacts],
            CA.[FTFContacts],

            -- ======================================================================== 
            -- DISCHARGE AND TREATMENT METRICS
            -- ========================================================================
            WT.[AverageLengthOfTreatment],
            WT.[AverageContactsAtDischarge],
            WT.[AverageFTFContactsAtDischarge],
            WT.[DischargesNoClockStop],
            WT.[DischargesWithClockStop]

        -- ============================================================================
        -- SUBQUERY 1: WAITING TIMES AND REFERRAL DATA
        -- ============================================================================
        FROM (
            SELECT
                WT.[ProviderCodeCurrent],
                SL.[Service_Line],
                WT.[PeriodEnd],

                -- --------------------------------------------------------------------
                -- New Referrals
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[RefStartThisPeriod] = 1 THEN WT.[Ref_ID] END) AS [Referrals],

                -- --------------------------------------------------------------------
                -- Clock Stop Activity
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[FirstContactInPeriod] = 1 THEN WT.[Ref_ID] END) AS [ClockStopActuals],

                -- --------------------------------------------------------------------
                -- Waiting List Metrics
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 1 AND WT.[RTTExclusion] = 0 THEN WT.[Ref_ID] END) AS [Waiters],
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 1 AND WT.[RTTExclusion] = 0 AND WT.[WaitingTime] > 7 * 18 THEN WT.[Ref_ID] END) AS [WaitersOver18Weeks],

                -- --------------------------------------------------------------------
                -- Current Caseload
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 0 AND WT.[WaitAssess] = 0 THEN WT.[Ref_ID] END) AS [Caseload],

                -- --------------------------------------------------------------------
                -- Discharge Quality Metrics
                -- --------------------------------------------------------------------
                AVG(CASE WHEN WT.[Discharged] = 1 THEN WT.[ContactsPerReferral] END) AS [AverageContactsAtDischarge],
                AVG(CASE WHEN WT.[Discharged] = 1 THEN WT.[ContactsFTFPerReferral] END) AS [AverageFTFContactsAtDischarge],
                AVG(CASE WHEN WT.[Discharged] = 1 THEN DATEDIFF(DAY, WT.[Ref_Start], WT.[FirstContact]) END) AS [AverageLengthOfTreatment],

                -- --------------------------------------------------------------------
                -- Discharge Breakdown
                -- --------------------------------------------------------------------
                COUNT(CASE WHEN WT.[Discharged] = 1 AND WT.[FirstContact] IS NULL THEN WT.[Ref_ID] END) AS [DischargesNoClockStop],
                COUNT(CASE WHEN WT.[Discharged] = 1 AND WT.[FirstContact] IS NOT NULL THEN WT.[Ref_ID] END) AS [DischargesWithClockStop]

            FROM (
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes]
			UNION  
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes_2324]
			) AS WT
            
			LEFT JOIN [MIS_Config].[dbo].[tbl_org_current_RL9_Service_Line] AS SL ON WT.[ProviderCodeCurrent] = SL.[Service_Codes]
			WHERE WT.[ProviderCodeCurrent] NOT IN ('996', '998')
				AND SL.[Status] = 'ACTIVE'
				AND SL.[RTT_Report_Enabled] = 1 -- RTT Reporting Only Services

            GROUP BY
                WT.[ProviderCodeCurrent],
                SL.[Service_Line],
                WT.[PeriodEnd]
                
        ) AS WT

        -- ============================================================================
        -- SUBQUERY 2: CONTACT ATTENDANCE DATA
        -- ============================================================================
        LEFT JOIN (
            SELECT
                CA.[ProviderCodeCurrent],
                CA.[PeriodEnd],

                -- --------------------------------------------------------------------
                -- Face-to-Face Contact Metrics
                -- --------------------------------------------------------------------
                -- FTF contacts after first contact (caseload activity)
                COUNT(CASE WHEN (CA.[FirstAttendance_FTF] = 1 OR CA.[FollowUp_FTF] = 1) AND CA.[Contact_Date] > WT.[FirstContact] THEN CA.[Ref_ID] END) AS [FTFCaseloadContacts],

                -- All FTF contacts (including first contacts)
                COUNT(CASE WHEN CA.[FirstAttendance_FTF] = 1 OR CA.[FollowUp_FTF] = 1 THEN CA.[Ref_ID] END) AS [FTFContacts],

                -- --------------------------------------------------------------------
                -- Total Contact Metrics
                -- --------------------------------------------------------------------
                -- Total contacts after first contact (caseload activity)
                COUNT(CASE WHEN CA.[PatientSeen] = 1 AND CA.[Contact_Date] > WT.[FirstContact] THEN CA.[Ref_ID] END) AS [TotalCaseloadContacts],

                -- All contacts where patient was seen
                COUNT(CASE WHEN CA.[PatientSeen] = 1 THEN CA.[Ref_ID] END) AS [TotalContacts]

            FROM (
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_ContactAttendances]
			UNION  
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_ContactAttendances_2324]
			) AS CA
            
			LEFT JOIN (
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes]
			UNION  
				SELECT * FROM [MIS_AG].[dbo].[Vw_tbl_ag_Report_WaitingTimes_2324]
			) AS WT
                ON CA.[Ref_ID] = WT.[Ref_ID] 
                --AND CA.[ProviderCodeCurrent] = WT.[ProviderCodeCurrent]
                AND CA.[PeriodEnd] = WT.[PeriodEnd]

            GROUP BY
                CA.[ProviderCodeCurrent],
                CA.[PeriodEnd]
                
        ) AS CA 
            ON WT.[ProviderCodeCurrent] = CA.[ProviderCodeCurrent] 
            AND WT.[PeriodEnd] = CA.[PeriodEnd]
                
        ORDER BY WT.[ProviderCodeCurrent], WT.[PeriodEnd]
		;