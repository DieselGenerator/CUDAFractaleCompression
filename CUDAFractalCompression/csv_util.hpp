/*
	csv“üo—Í‚ğ’S‚¤
*/

#pragma once

#include <vector>
#include <string>

#include "ifs_transform_data.hpp"

void import_csv(std::string, std::vector<ifs_transformer*>&);
void export_csv(std::string, std::vector<ifs_transformer*>&, long long, long long, double);
