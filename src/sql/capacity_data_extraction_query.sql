		-- ============================================================================
        -- STAFFING DATA EXTRACTION QUERY
        -- ============================================================================
        -- Purpose: Extract staffing capacity metrics from ESR appraisal review data,
        -- pulling staff counts by provider, service line, and staff group.
        -- ============================================================================
        SELECT
            COUNT(DISTINCT ARD.[Assignment Number]) AS [Staff],
            ARD.[Staff Group],

            -- ======================================================================== 
            -- PROVIDER CODE EXTRACTION
            -- ========================================================================
            -- Provider code: the 3 digits after 'L5 '
            SUBSTRING(ARD.[Org L6], CHARINDEX('L5 ', ARD.[Org L6]) + 3, 3) AS [ProviderCodeCurrent],
			SL.[Service_Line]

        FROM [ISEVSQLMIS-BLK].[ESR].[dbo].[tbl_dt_Appraisal_Review_Detail] AS ARD
		LEFT JOIN [MIS_Config].[dbo].[tbl_org_current_RL9_Service_Line] AS SL ON SUBSTRING(ARD.[Org L6], CHARINDEX('L5 ', ARD.[Org L6]) + 3, 3) = SL.[Service_Codes]
		
		WHERE SL.[Status] = 'ACTIVE'
		AND SL.[RTT_Report_Enabled] = 1 -- RTT Reporting Only Services

        GROUP BY
            ARD.[Org L6],
			SL.[Service_Line],
            ARD.[Staff Group]

        ORDER BY
            ARD.[Org L6],
            SL.[Service_Line],
            ARD.[Staff Group];