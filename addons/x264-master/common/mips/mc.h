/*****************************************************************************
 * mc.h: msa motion compensation
 *****************************************************************************
 * Copyright (C) 2015-2023 x264 project
 *
 * Authors: Neha Rana <neha.rana@imgtec.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/

#ifndef X264_MIPS_MC_H
#define X264_MIPS_MC_H

#define x264_mc_init_mips x264_template(mc_init_mips)
void x264_mc_init_mips( uint32_t cpu, x264_mc_functions_t *pf );

#endif
