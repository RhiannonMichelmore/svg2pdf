from xml.dom import minidom
import drawSvg as draw
import sys
from shapely.geometry import LineString, LinearRing, Point, Polygon
import shapely
import numpy as np
import matplotlib.pyplot as plt
import os
import math

def is_in_range(colour,target,rad):
    """ Takes a colour, a colour target (in string of hex form e.g '#000000') and a distance (int)
        and tells you whether the colour is within or equal to distance of the target.
    """
    if len(colour) == 7:
        r = int('0x'+colour[1:3],16)
        g = int('0x'+colour[3:5],16)
        b = int('0x'+colour[5:],16)
        
        rt = int('0x'+target[1:3],16)
        gt = int('0x'+target[3:5],16)
        bt = int('0x'+target[5:],16)

        if abs(r-rt) <= rad and abs(g-gt) <= rad and abs(b-bt) <= rad:
            return True
        else:
            return False
    else:
        return False
    
def matrix_transform(point_list,params):
    """Applies matrix transform on a list of points [(x,y)] as specified in the SVG standard."""
    new_points = []
    for point in point_list:
        old_x = point[0]
        old_y = point[1]
        new_x = (params[0]*old_x) + (params[2]*old_y) + params[4]
        new_y = (params[1]*old_x) + (params[3]*old_y) + params[5]
        new_points.append((new_x,new_y))
    return new_points

def scale_transform(point_list,params):
    """Applies scale transform on a list of points [(x,y)] as specified in the SVG standard."""
    if len(params) == 1:
        sx = params[0]
        sy = params[0]
    else:
        sx = params[0]
        sy = params[1]
        
    new_points = []
    for point in point_list:
        old_x = point[0]
        old_y = point[1]
        new_x = old_x*sx
        new_y = old_y*sy
        new_points.append((new_x,new_y))
    return new_points

def transform_points(idx,transforms,points,order):
    """ Applies given transforms to a list of points, in a given order.
    
        Order is taken from the following master indices:
            [scale=0, matrix=1]
            
        E.g order = [1,0] means matrix is applied, then scale.
    """
    has_transform = [False for i in range(2)]
    
    for i,v in enumerate(master_transform_order):
        tmp = transforms[v][idx]
        if not tmp == None:
            has_transform[i] = True
            
    for t in order:
        if has_transform[t]:
            points = master_transform_functions[t](points,transforms[master_transform_order[t]][idx])
            
    return points
            

def extract_paths(path_strings, path_styles, colour, transforms, transforms_order_list, return_styles=False):
    """ This function does the following:
        1) Extracts all paths of a given colour (within a range of 3 hex values of that colour)
        2) Converts all curves to straight lines
        3) Converts all points in the path to absolute values
        4) Applies all given transforms in a given order
        5) Returns all the paths in [[(x,y)],[(x,y)]] form and (optionally) the style string of each path 
    """
    colour_path_strings = []
    colour_path_styles = []
    transform_idxs = []

    for i,p in enumerate(path_strings):
        if (colour == 'none' and path_styles[i]['stroke'].lower() == colour) or (not colour == 'none' and is_in_range(path_styles[i]['stroke'].lower(),colour,3)):
            colour_path_strings.append(p)
            colour_path_styles.append(path_styles[i])
            transform_idxs.append(i)
            
    colour_paths = []
    token_types = ['M','m','L','l','V','v','H','h','C','c']
    for entry,cps in enumerate(colour_path_strings):
        transform_idx = transform_idxs[entry] 
        transforms_order = transforms_order_list[transform_idxs[entry]]
        tokens = cps.split(' ')
        idx = 0
        points = []
        m_count = 0
        mode = None
        while idx < len(tokens):
            current_token = tokens[idx]
            point = None
            if current_token in token_types:
                mode = current_token
                # deals with the case where we have a lowercase m at the start: first coord treated as absolute
                if current_token == 'm' and m_count == 0:
                    first = True
                # a path can have multiple m's in it, and we want to submit the current line if we find another
                # m after the first
                if current_token == 'm' or current_token == 'M':
                    m_count += 1
                    if m_count > 1:
                        last_point = points[-1]
                        points = transform_points(transform_idx,transforms,points,transforms_order)
                        points = [(x,-1*y) for (x,y) in points]
                        colour_paths.append(points)
                        points = []
                idx += 1
            elif current_token == 'z' or current_token == 'Z':
                point = points[0]
                idx += 1
            elif not current_token[0].isalpha():
                if mode == 'M' or mode == 'L':
                    x_val = float(tokens[idx].split(',')[0])
                    y_val = float(tokens[idx].split(',')[1])
                    point = (x_val,y_val)
                    idx += 1
                elif mode == 'm':
                    x_val = float(tokens[idx].split(',')[0])
                    y_val = float(tokens[idx].split(',')[1])
                    if first == False:
                        if len(points) == 0:
                            point = (x_val+last_point[0],y_val+last_point[1])
                        else:
                            point = (x_val+points[-1][0],y_val+points[-1][1])

                    else:
                        first = False
                        point = (x_val,y_val)
                    idx += 1
                elif mode == 'l':
                    x_val = float(tokens[idx].split(',')[0])
                    y_val = float(tokens[idx].split(',')[1])
                    point = (x_val+points[-1][0],y_val+points[-1][1])
                    idx += 1
                elif mode == 'V':
                    x_val = points[-1][0]
                    y_val = float(tokens[idx])
                    new_y = y_val
                    new_x = x_val
                    point = (new_x,new_y)
                    idx += 1
                elif mode == 'v':
                    x_val = points[-1][0]
                    y_val = float(tokens[idx])
                    new_y = y_val + points[-1][1]
                    new_x = x_val
                    point = (new_x,new_y)
                    idx += 1
                elif mode == 'H':
                    x_val = float(tokens[idx])
                    y_val = points[-1][1]
                    new_x = x_val
                    new_y = y_val
                    point = (new_x,new_y)
                    idx += 1
                elif mode == 'h':
                    x_val = float(tokens[idx])
                    y_val = points[-1][1]
                    new_x = x_val + points[-1][0]
                    new_y = y_val
                    point = (new_x,new_y)
                    idx += 1
                elif mode == 'C':
                    end_point = tokens[idx+2]
                    x_val = float(end_point.split(',')[0])
                    y_val = float(end_point.split(',')[1])
                    point = (x_val,y_val)
                    idx += 3
                elif mode == 'c':
                    end_point = tokens[idx+2]
                    x_val = float(end_point.split(',')[0])
                    y_val = float(end_point.split(',')[1])
                    point = (x_val+points[-1][0],y_val+points[-1][1])
                    idx += 3     
            else:
                print('unsupported token type')
                
            if not point == None:
                points.append(point)
            
        points = transform_points(transform_idx,transforms,points,transforms_order)
        points = [(x,-1*y) for (x,y) in points]
        colour_paths.append(points)
        
    if return_styles:
        return colour_paths, colour_path_styles
    else:
        return colour_paths

# takes two (x,y) pairs and return True if within threshold of closeness in both x and y
def app_eq(c1,c2,threshold=2):
    """ Takes two points and returns True if they are within threshold 
        distance to eachother, in both the x and y axis.
    """
    if abs(c1[0]-c2[0]) < threshold and abs(c1[1]-c2[1]) < threshold:
        return True
    else:
        return False
    
def enlarge(factor,line):
    """ Takes a line of the form ((x1,y1),(x2,y2)) and returns a line lengthened by factor."""
    p1 = line[0]
    p2 = line[1]
    t0=0.5*(1.0-factor)
    t1=0.5*(1.0+factor)
    x1 = p1[0] +(p2[0] - p1[0]) * t0
    y1 = p1[1] +(p2[1] - p1[1]) * t0
    x2 = p1[0] +(p2[0] - p1[0]) * t1
    y2 = p1[1] +(p2[1] - p1[1]) * t1
    return ((x1,y1),(x2,y2))

#TODO: I THINK THIS DOESN'T WORK QUITE RIGHT, RETHINK IT (can have false positives if lines are paralell and close?)
def contained_within(line1,line2):
    """ Checks whether line1 is completely within line2, with some ease (limit is hard coded to 0.05 rad)."""
    #checks whether line1 is completely within line2, with some ease
    line1_vec = (line1[1][0]-line1[0][0],line1[1][1]-line1[0][1])
    line2_vec = (line2[1][0]-line2[0][0],line2[1][1]-line2[0][1])
    angle = angle_between(line1_vec,line2_vec)
    limit = 0.05
    if -limit <= angle and angle <= limit:
        #print('potential candidate')
        if line1[0][0] >= line2[0][0] and line1[1][0] <= line2[1][0] and line1[0][1] >= line2[0][1] and line1[1][1] <= line2[1][1]:
            return True
        else:
            return False
    else:
        return False

def split_paths(colour_paths):     
    # new idea: just get all lines and remove intersections
    lines = []
    for p in colour_paths:
        for i, coords in enumerate(p):
            if i < len(p)-1:
                lines.append((coords,p[i+1]))

    new_lines = []
    no_subs = False
    while not no_subs:
        for i in range(len(lines)):
            for j in range(i+1,len(lines)):
                line1 = lines[i]
                line2 = lines[j]
                sf = 1.03
                line1_s = enlarge(sf,lines[i])
                line2_s = enlarge(sf,lines[j])
                LS1 = LineString([line1_s[0], line1_s[1]])
                LS2 = LineString([line2_s[0], line2_s[1]])
                its = LS1.intersection(LS2)
                if type(its) == shapely.geometry.linestring.LineString:
                    # if line1 is totally contained within line 2, we want to delete line1
                    if contained_within(line1,line2):
                        lines = lines[:i] + lines[i+1:]
                        found_sub = True
                        break
                    continue
                found_sub = False
                if its:
                    if app_eq(line1[0],line2[0]) or app_eq(line1[0],line2[1]) or app_eq(line1[1],line2[0]) or app_eq(line1[1],line2[1]):
                            # shared start/end point, not true intersection
                            pass
                    else:
                        # need to see if its the end of one line and the middle of the other
                        to_check = (its.x,its.y)
                        if app_eq(to_check,line1[0]) or app_eq(to_check,line1[1]):
                            # its either at the start or end of line 1 so only need to break line 2
                            changed_lines = [line1,(line2[0],to_check),(to_check,line2[1])]
                        elif app_eq(to_check,line2[0]) or app_eq(to_check,line2[1]):
                            # its either at the start or end of line 2 so only need to break line 1
                            changed_lines = [line2,(line1[0],to_check),(to_check,line1[1])]
                        else:
                            # need to break both
                            changed_lines = [(line1[0],to_check),(to_check,line1[1]),(line2[0],to_check),(to_check,line2[1])]

                        lines = lines[:i] + lines[i+1:j] + lines[j+1:] + changed_lines
                        found_sub = True
                        break

            if found_sub == True:
                break
        if found_sub == False:
            no_subs = True

    #remove duplicate lines
    lines = list(set(lines))
    return lines

def angle_between(vector1, vector2):
    """ Returns the angle in radians between given vectors."""
    v1_u = unit_vector(vector1)
    v2_u = unit_vector(vector2)
    minor = np.linalg.det(
        np.stack((v1_u[-2:], v2_u[-2:]))
    )
    if minor == 0:
        sign = 1
    else:
        sign = -np.sign(minor)
    dot_p = np.dot(v1_u, v2_u)
    dot_p = min(max(dot_p, -1.0), 1.0)
    return -1 * sign * np.arccos(dot_p)

def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def check_same_lines(poly1,poly2):
    """ Checks whether two polygons are identical, 
        accounting for the fact that some lines may be defined backwards.
    """
    same = True
    for line in poly1:
        if line in poly2 or (line[1],line[0]) in poly2:
            pass
        else:
            same = False
    return same

def parse_transforms_string(transforms_string):
    c = 0
    ordering = []
    current_transform = dict()
    string_fragment = transforms_string
    while not string_fragment.find('(') == -1:
        start_idx = string_fragment.find('(')
        end_idx = string_fragment.find(')')
        transformation_type = string_fragment[:start_idx]
        if not transformation_type in master_transform_order:
            print('Unsupported transformation type:',transformation_type)
            return
        params = string_fragment[start_idx+1:end_idx].strip()
        if ',' in params:
            params = [float(p.strip()) for p in params.split(',')]
        else:
            params = [float(p.strip()) for p in params.split(' ')]
            
        current_transform[transformation_type] = params
        ordering.append(master_transform_order.index(transformation_type))
        
        if end_idx + 1 < len(string_fragment):
            string_fragment = string_fragment[end_idx+1:].lstrip()
        else:
            string_fragment = ''
            
    return current_transform,ordering

def parse_transforms_strings(transforms_list):
    """ Takes a list of transform strings (corresponding to paths), and returns a
        transforms dict with key as transform name and value as a list of transform parameters (or none if
        no transform of that type for the current path), along
        with a list of lists of which transforms are applied in what order.
    """
    transforms = dict()
    for key in master_transform_order:
        transforms[key] = []

    orderings = []
    # separators can be space OR comma
    for t in transforms_list:
        current_transform,current_ordering = parse_transforms_string(t)
        for key in master_transform_order:
            if key in current_transform:
                transforms[key].append(current_transform[key])
            else:
                transforms[key].append(None)
        orderings.append(current_ordering)
    return transforms, orderings



def main(svg_file):
    
    print('Parsing file:',svg_file)
    
    doc = minidom.parse(svg_file)
    g = doc.getElementsByTagName('g')[0]
    transform = g.getAttribute('transform')
    if not transform == '':
        if len(transform.split(')')) > 2:
            print('Only support translation on the whole SVG at the moment.')
        if not transform.split('(')[0] == 'translate':
            print('Only supports translation on the whole SVG at the moment.')
        coords = transform.split('(')[1][:-1]
        dx = float(coords.split(',')[0])
        dy = float(coords.split(',')[1])
    else:
        dx = 0
        dy = 0

    print('Extracting path strings.')
    path_strings = [(path.getAttribute('style'),path.getAttribute('d'),path.getAttribute('transform')) for path
                    in g.getElementsByTagName('path')]

    print('Extracting text strings.')
    text_strings = []
    for text in g.getElementsByTagName('text'):
        sty = text.getAttribute('style')
        x = float(text.getAttribute('x'))
        y = -1*(float(text.getAttribute('y')))
        if not text.firstChild.firstChild == None:
            string = text.firstChild.firstChild.nodeValue
            text_strings.append((sty,x,y,string))

    width = doc.getElementsByTagName('svg')[0].getAttribute('width')
    height = doc.getElementsByTagName('svg')[0].getAttribute('height')
    if width.endswith('mm'):
        width = round(float(width[:-2]),0)
    else:
        print('unknown width unit')
        sys.exit(0)

    if height.endswith('mm'):
        height = round(float(height[:-2]),0)
    else:
        print('unknown height unit')
        sys.exit(0)

    doc.unlink()
    
    path_styles = []
    paths = []
    transforms = []
    for (s,p,t) in path_strings:
        style_map = dict()
        styles = s.split(';')
        for s_string in styles:
            spl = s_string.split(':')
            style_map[spl[0]] = spl[1]
        path_styles.append(style_map)
        paths.append(p)
        transforms.append(t)
        
    print('Parsing transforms.')
    transforms, orderings = parse_transforms_strings(transforms)
    
    
    print('Extracting red segment paths.')
    red_paths = extract_paths(paths,path_styles,'#ff0000',transforms,orderings)
    print('Splitting overlapping red paths.')
    red_lines = split_paths(red_paths)
    red_lines = red_lines + [(l[1],l[0]) for l in red_lines]
    
    print('Creating red segment polygons.')
    polygons = []
    for line in red_lines:
        found = False
        current_polygon = [line]
        used_lines = [line,(line[1],line[0])]
        current_line = line

        while not found:
        # first get all lines that connect to point 2
            current_line = current_polygon[-1]
            line_start = current_line[0]
            line_end = current_line[1]
            line_vec = (line_end[0]-line_start[0],line_end[1]-line_start[1])
            connecting_lines = [l for l in red_lines if (app_eq(l[0],line_end) or app_eq(l[1],line_end)) and not l in used_lines]
            #print(connecting_lines)
            max_angle = -10000
            chosen_line = None        
            for cl in connecting_lines:
                cl_end = cl[0] if app_eq(line_end,cl[1]) else cl[1]
                cl_start = cl[1] if app_eq(line_end,cl[1]) else cl[0]
                new_cl = (cl_start,cl_end)
                cl_vec = (cl_end[0]-cl_start[0],cl_end[1]-cl_start[1])
                angle = angle_between(line_vec,cl_vec)
                if angle > max_angle:
                    max_angle = angle
                    chosen_line = new_cl

            current_polygon.append(chosen_line)
            used_lines.append(chosen_line)
            used_lines.append((chosen_line[1],chosen_line[0]))

            if app_eq(chosen_line[1],current_polygon[0][0]):
                found = True
                polygons.append(current_polygon)
     
    print('Filtering duplicate segments.')
    unique_polygons = [polygons[0]]
    for i,p in enumerate(polygons):
        same = False
        for j,q in enumerate(unique_polygons):
                if check_same_lines(p,q):
                    same = True
        if same == False:
            unique_polygons.append(p)
            
    print('Extracting aligment marker paths.')
    blue_paths = extract_paths(paths,path_styles,'#0000ff',transforms,orderings)
    print('Extracting all black paths.')
    black_paths = extract_paths(paths,path_styles,'#000000',transforms,orderings)
    print('Splitting overlapping black paths.')
    black_lines = split_paths(black_paths)

    print('Extracting fill area paths.')
    fill_paths, fill_path_styles = extract_paths(paths,path_styles,'none',transforms,orderings,True)
    
    print('Converting all paths to Shapely lines or polygons.')
    black_ls = []
    for line in black_lines:
        black_ls.append(LineString([Point(line[0][0],line[0][1]),Point(line[1][0],line[1][1])]))

    blue_ls = []
    for line in blue_paths:
        blue_ls.append(LineString([Point(line[0][0],line[0][1]),Point(line[1][0],line[1][1])]))

    text_points = []
    for ts in text_strings:
        # (style,x,y,value)
        text_points.append(Point(ts[1],ts[2]))

    fill_polys = []
    for points in fill_paths:
        fill_polys.append(Polygon([Point(x,y) for (x,y) in points]))

    shapely_polys = []
    for polygon in unique_polygons:
        poly_points = []
        for i,line in enumerate(polygon):
            if len(poly_points) == 0:
                poly_points.append(Point(line[0][0],line[0][1]))
            if i < len(polygon)-1:
                poly_points.append(Point(line[1][0],line[1][1]))
        poly_points.append(Point(polygon[0][0][0],polygon[0][0][1]))      
        lr = Polygon(poly_points)
        shapely_polys.append(lr)

    svgs = []

    print('Drawing each segment, lines and fill, and converting to SVG.')
    for q,polygon in enumerate(shapely_polys):
        outset = 6
        origin = (-dx-outset,-(height+outset+1)+dy)
        d = draw.Drawing(width+1+2*outset,height+2+2*outset,origin=origin)
        d.setRenderSize(str(width+(2*6.35))+'mm',str(height+(2*6.35))+'mm')

        lr = polygon
        expand_lr = lr.buffer(1.5, resolution=16, join_style=2, mitre_limit=30).exterior
        expand_lr = Polygon(expand_lr.coords)

        # compute seam allowance, dist 6 == 1/4 inch
        outset = lr.buffer(6,resolution=16,join_style=2,mitre_limit=45).exterior
        outset = [(l[0],l[1]) for l in outset.coords]

        biggest = True
        for r,other_poly in enumerate(shapely_polys):
            if not expand_lr.contains(other_poly):
                biggest = False
        if biggest:
            continue

        #plt.plot(*lr.exterior.xy,c='b')
        #plt.plot(*expand_lr.xy,c='r')
        within_idxs = []
        for i,bl in enumerate(black_ls):
            if expand_lr.contains(bl):
                within_idxs.append(i)

        within_lines = [x for i,x in enumerate(black_lines) if i in within_idxs]

        marker_idxs = []
        for i,bl in enumerate(blue_ls):
            if lr.intersects(bl):
                marker_idxs.append(i)

        align_lines = [x for i,x in enumerate(blue_paths) if i in marker_idxs]

        text_idxs = []
        for i,tp in enumerate(text_points):
            if lr.contains(tp):
                text_idxs.append(i)
        text_locations = [x for i,x in enumerate(text_strings) if i in text_idxs]

        fill_idxs = []
        for i,fp in enumerate(fill_polys):
            if expand_lr.contains(fp):
                fill_idxs.append(i)

        fill_poly_paths = [x for i,x in enumerate(fill_paths) if i in fill_idxs]
        fill_poly_styles = [x for i,x in enumerate(fill_path_styles) if i in fill_idxs]

        for i,points in enumerate(fill_poly_paths):
            colour = fill_poly_styles[i]['fill']
            p = draw.Path(stroke_width=0,fill=colour,fill_opacity=1)
            p.M(points[0][0],points[0][1])
            for point in points[1:]:
                p.L(point[0],point[1])
            d.append(p)
        for points in unique_polygons[q]:
            p = draw.Path(stroke_width=3,stroke='red',fill_opacity=0,opacity=0.3)
            p.M(points[0][0],points[0][1])
            for point in points[1:]:
                p.L(point[0],point[1])
            d.append(p)
        for points in within_lines:
            p = draw.Path(stroke_width=1,stroke='black',fill_opacity=0)
            p.M(points[0][0],points[0][1])
            for point in points[1:]:
                p.L(point[0],point[1])
            d.append(p)
        for points in align_lines:
            p = draw.Path(stroke_width=1,stroke='blue',fill_opacity=0)
            p.M(points[0][0],points[0][1])
            for point in points[1:]:
                p.L(point[0],point[1])
            d.append(p)
        for ts in text_locations:
            size = float([x.split(':')[1] for x in ts[0].split(';') if x.split(':')[0] == 'font-size'][0][:-2])
            colour = [x.split(':')[1] for x in ts[0].split(';') if x.split(':')[0] == 'fill'][0]
            p = draw.Text(ts[3],size,float(ts[1]),float(ts[2]),fill=colour)
            if ts[3].isalpha():
                letter = ts[3]
            d.append(p)

        # add seam allowance
        p = draw.Path(stroke='black',stroke_dasharray=3,fill_opacity=0,opacity=1)
        p.M(outset[0][0],outset[0][1])
        for point in outset[1:]:
            p.L(point[0],point[1])
        d.append(p)
        svgs.append((d,letter))

    print("Saving generated segment SVG's to file.")
    directory = svg_file[:-4]
    if not os.path.exists(directory):
        os.mkdir(directory)
    for svg,letter in svgs:
        svg.saveSvg(directory+'/'+letter+'.svg')
        
    print('Done.')


if __name__ == '__main__':

    master_transform_order = ['scale','matrix']
    master_transform_functions = [scale_transform,matrix_transform]

    if not len(sys.argv) == 2:
        print("Usage: python extract.py <svg_file_path>")
        sys.exit(0)

    main(sys.argv[1])
